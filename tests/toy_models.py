'''测试用玩具模型：forward 签名和返回的 .logits 与 HF CausalLM 一致，并记录调用次数。'''

import math
from types import SimpleNamespace

import torch


class ContextFreeModel(torch.nn.Module):
    '''每个位置都输出同一组 logits，与上下文无关。'''

    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer('fixed_logits', logits.float())
        self.calls = 0

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
        self.calls += 1
        batch, seq_len = input_ids.shape
        return SimpleNamespace(logits=self.fixed_logits.expand(batch, seq_len, -1))


class BigramModel(torch.nn.Module):
    '''下一个 token 的 logits 只取决于当前 token，用来覆盖依赖上下文的情况。'''

    def __init__(self, vocab_size: int, seed: int):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer('table', 3 * torch.randn(vocab_size, vocab_size, generator=generator))
        self.calls = 0

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
        self.calls += 1
        return SimpleNamespace(logits=self.table[input_ids])


class CallCounter(torch.nn.Module):
    '''包一层真实模型，只统计 forward 次数。'''

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.calls = 0

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False):
        self.calls += 1
        return self.model(input_ids, use_cache=use_cache)


def greedy_reference(model: torch.nn.Module, prompt: torch.Tensor, n_new_tokens: int):
    '''不依赖 main.py 的 greedy 参考实现。'''
    x = prompt
    with torch.no_grad():
        for _ in range(n_new_tokens):
            logits = model(x, use_cache=False).logits[:, -1, :]
            x = torch.cat((x, logits.argmax(dim=-1, keepdim=True)), dim=1)
    return x


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
