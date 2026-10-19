'''
测试用玩具模型：forward 的参数和返回值与 HF CausalLM 一致(input_ids、past_key_values、
logits_to_keep → .logits / .past_key_values),并记录调用次数、喂入 token 数和 lm_head 输出行数
'''

import math
from types import SimpleNamespace

import torch
from transformers import DynamicCache


class ToyCausalLM(torch.nn.Module):
    '''
    子类只需实现 full_logits(tokens):给定完整序列，返回每个位置的 logits
    传入 DynamicCache 时，把已见过的 token id 当作 key 存进 cache,
    这样 crop / get_seq_length 走的是 transformers 自己的实现
    '''

    def __init__(self):
        super().__init__()
        self.calls = 0
        self.tokens_fed = 0
        self.logit_rows = []

    def reset_stats(self) -> None:
        self.calls = 0
        self.tokens_fed = 0
        self.logit_rows = []

    def full_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: DynamicCache | None = None,
        use_cache: bool = False,
        logits_to_keep: int = 0,
        **kwargs,
    ):
        self.calls += 1
        self.tokens_fed += input_ids.shape[1]

        history = input_ids
        if past_key_values is not None:
            ids = input_ids.float()[:, None, :, None]
            keys, _ = past_key_values.update(ids, ids, layer_idx=0)
            history = keys[:, 0, :, 0].long()

        logits = self.full_logits(history)[:, -input_ids.shape[1] :, :]
        if logits_to_keep > 0:
            logits = logits[:, -logits_to_keep:, :]
        self.logit_rows.append(logits.shape[1])
        return SimpleNamespace(logits=logits, past_key_values=past_key_values)


class ContextFreeModel(ToyCausalLM):
    '''每个位置都输出同一组 logits，与上下文无关。'''

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer('fixed_logits', logits.float())

    def full_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq_len = tokens.shape
        return self.fixed_logits.expand(batch, seq_len, -1)


class BigramModel(ToyCausalLM):
    '''下一个 token 的 logits 只取决于当前 token。'''

    def __init__(self, vocab_size: int, seed: int):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer('table', 3 * torch.randn(vocab_size, vocab_size, generator=generator))

    def full_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.table[tokens]


class PrefixSumModel(ToyCausalLM):
    '''
    logits 取决于整个前缀的 token 之和(mod vocab):
    回滚后 cache 里只要残留一个被拒绝的草稿，后面每个位置的输出都会变
    '''

    def __init__(self, vocab_size: int, seed: int):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer('table', 3 * torch.randn(vocab_size, vocab_size, generator=generator))

    def full_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.table[tokens.cumsum(dim=-1) % self.table.shape[0]]


class CallCounter(torch.nn.Module):
    '''包一层真实模型，只统计 forward 次数，其余参数原样透传。'''

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.calls = 0

    def forward(self, input_ids: torch.Tensor, **kwargs):
        self.calls += 1
        return self.model(input_ids, **kwargs)


def greedy_reference(model: torch.nn.Module, prompt: torch.Tensor, n_new_tokens: int):
    '''不依赖 main.py 的 greedy 参考实现。'''
    x = prompt
    with torch.no_grad():
        for _ in range(n_new_tokens):
            logits = model(x, use_cache=False).logits[:, -1, :]
            x = torch.cat((x, logits.argmax(dim=-1, keepdim=True)), dim=1)
    return x


def greedy_speculative_reference(
    draft: torch.nn.Module,
    target: torch.nn.Module,
    prompt: torch.Tensor,
    n_new_tokens: int,
    gamma: int,
) -> tuple[torch.Tensor, int, int]:
    '''
    不带 KV cache、每次整段重算的 greedy 投机解码
    返回 (输出序列, 目标模型调用次数, 草稿模型调用次数)
    '''
    x = prompt
    target_len = prompt.shape[1] + n_new_tokens
    target_calls = draft_calls = 0
    with torch.no_grad():
        while x.shape[1] < target_len:
            prefix_len = x.shape[1]
            k = min(gamma, target_len - prefix_len - 1)
            drafted = x
            for _ in range(k):
                logits = draft(drafted, use_cache=False).logits[:, -1, :]
                drafted = torch.cat((drafted, logits.argmax(dim=-1, keepdim=True)), dim=1)
                draft_calls += 1
            target_logits = target(drafted, use_cache=False).logits[0, prefix_len - 1 :, :]
            target_pred = target_logits.argmax(dim=-1)
            target_calls += 1
            n_accepted = 0
            while n_accepted < k and drafted[0, prefix_len + n_accepted] == target_pred[n_accepted]:
                n_accepted += 1
            next_token = target_pred[n_accepted].view(1, 1)
            x = torch.cat((drafted[:, : prefix_len + n_accepted], next_token), dim=1)
    return x, target_calls, draft_calls


def tv_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return 0.5 * (a - b).abs().sum().item()


def chi_square_critical(df: int, z: float = 3.0902) -> float:
    '''Wilson-Hilferty 近似的卡方分位数，默认 z 对应显著性 1e-3。'''
    c = 2 / (9 * df)
    return df * (1 - c + z * math.sqrt(c)) ** 3


def assert_samples_follow(counts: torch.Tensor, probs: torch.Tensor, tv_tol: float) -> None:
    '''检查经验频数 counts 是否服从 probs：零概率处必须零计数，再做卡方检验和 TV 检查。'''
    counts = counts.double()
    probs = probs.double()
    n = counts.sum()
    support = probs > 0

    leaked = counts[~support].sum().item()
    assert leaked == 0, f'{leaked:.0f} samples landed on zero-probability tokens'

    expected = probs[support] * n
    chi2 = ((counts[support] - expected) ** 2 / expected).sum().item()
    df = int(support.sum().item()) - 1
    critical = chi_square_critical(df)
    assert chi2 < critical, f'chi2 {chi2:.2f} >= critical {critical:.2f} (df={df})'

    tv = tv_distance(counts / n, probs)
    assert tv < tv_tol, f'TV {tv:.4f} >= {tv_tol}'
