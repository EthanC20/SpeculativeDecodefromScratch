import argparse
import time
from collections.abc import Callable
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# 草稿模型与目标模型必须共用同一套词表，否则无法逐 token 比较概率
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_APPROX_MODEL = PROJECT_ROOT / 'hf_models' / 'Qwen2.5-0.5B-Instruct'
DEFAULT_TARGET_MODEL = PROJECT_ROOT / 'hf_models' / 'Qwen2.5-3B-Instruct'


def parse_arguments():
    parser = argparse.ArgumentParser(description='Autoregressive vs speculative sampling')
    parser.add_argument('--input', type=str, default='A dog has four ')
    parser.add_argument('--approx_model_name', type=str, default=str(DEFAULT_APPROX_MODEL))
    parser.add_argument('--target_model_name', type=str, default=str(DEFAULT_TARGET_MODEL))
    parser.add_argument('--num_tokens', type=int, default=20)
    parser.add_argument('--gamma', type=int, default=4)
    parser.add_argument('--temperature', type=float, default=1.0, help='0 means greedy decoding')
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def synchronize(device: torch.device) -> None:
    # MPS 是异步提交，不同步的话测到的只是提交时间
    if device.type == 'mps':
        torch.mps.synchronize()


def timed(fn: Callable[[], torch.Tensor], device: torch.device) -> tuple[torch.Tensor, float]:
    synchronize(device)
    start = time.perf_counter()
    out = fn()
    synchronize(device)
    return out, time.perf_counter() - start


def top_k_top_p_filter(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0) -> torch.Tensor:
    '''把不在 top-k / top-p 核内的 logit 置为 -inf,原地修改传入张量。'''
    if top_k > 0:
        threshold = torch.topk(logits, min(top_k, logits.size(-1)))[0]
        logits[logits < threshold[:, [-1]]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # 右移一位，保证至少保留第一个使累计概率越过 top_p 的 token
        remove_mask = cumulative_probs > top_p
        remove_mask[..., 1:] = remove_mask[..., :-1].clone()
        remove_mask[..., 0] = 0
        indices_to_remove = remove_mask.scatter(1, sorted_indices, remove_mask)
        logits[indices_to_remove] = float('-inf')
    return logits


def norm_logits(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    '''
    把 (batch, vocab) logits 变成实际采样所用的概率 p' / q'
    草稿采样、验证、残差和 bonus 都必须用同一个变换，否则投机采样不再无损
    temperature=0 表示 greedy,返回 argmax 的 one-hot
    权重是 bfloat16,这里统一升到 float32,避免 MPS 上概率比较失真
    '''
    logits = logits.float()
    if temperature == 0:
        return F.one_hot(logits.argmax(dim=-1), num_classes=logits.size(-1)).float()
    logits = top_k_top_p_filter(logits / temperature, top_k=top_k, top_p=top_p)
    return F.softmax(logits, dim=-1)


def sample(probs: torch.Tensor) -> torch.Tensor:
    '''从 (batch, vocab) 概率中抽一个 token,返回形状 (batch, 1)。'''
    return torch.multinomial(probs, num_samples=1)


class KVCacheModel:
    '''
    用 DynamicCache 记住已经算过的前缀，每次只把 cache 之后的新 token 喂给模型
    lm_head 只对最后 logits_to_keep 个位置计算
    '''

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.cache = DynamicCache()

    def __call__(self, x: torch.Tensor, logits_to_keep: int) -> torch.Tensor:
        '''x 是完整序列，返回最后 logits_to_keep 个位置的 logits,形状 (batch, logits_to_keep, vocab)'''
        new_ids = x[:, self.cache.get_seq_length() :]
        assert new_ids.shape[1] >= logits_to_keep, (
            f'only {new_ids.shape[1]} uncached tokens, cannot keep {logits_to_keep} logits'
        )
        out = self.model(
            new_ids,
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=logits_to_keep,
        )
        return out.logits

    def rollback(self, length: int) -> None:
        '''丢掉位置 length 及之后的 KV;cache 本来就更短时不变'''
        self.cache.crop(length)


def validate_sampling_args(temperature: float, top_p: float, gamma: int = 1) -> None:
    if temperature < 0:
        raise ValueError('temperature must be >= 0')
    if not 0.0 <= top_p <= 1.0:
        raise ValueError('top_p must be in [0, 1]')
    if gamma < 1:
        raise ValueError('gamma must be >= 1')


def autoregressive_sampling(
    x: torch.Tensor,
    model: torch.nn.Module,
    n_new_tokens: int,
    temperature: float = 1,
    top_k: int = 0,
    top_p: float = 0,
) -> torch.Tensor:
    '''只用 target model 逐 token 生成,作为投机采样的对照。'''
    validate_sampling_args(temperature, top_p)
    cached_model = KVCacheModel(model)
    generated = 0
    with torch.no_grad(), tqdm(total=n_new_tokens, desc='autoregressive sampling') as pbar:
        while generated < n_new_tokens:
            logits = cached_model(x, logits_to_keep=1)[:, -1, :]
            next_token = sample(norm_logits(logits, temperature, top_k, top_p))
            x = torch.cat((x, next_token), dim=1)
            generated += 1
            pbar.update(1)
    return x


def max_fn(x: torch.Tensor) -> torch.Tensor:
    '''norm(max(x, 0)):丢掉非正项后按行重新归一化'''
    x_max = torch.where(x > 0, x, torch.zeros_like(x))
    return x_max / torch.sum(x_max, dim=-1, keepdim=True)


def verify_draft(
    p: torch.Tensor,
    q: torch.Tensor,
    draft_tokens: torch.Tensor,
) -> tuple[int, torch.Tensor]:
    '''
    对 k 个草稿 token 逐个做拒绝采样
    p: (k + 1, vocab) 目标模型在每个草稿位置及其后一位的处理后概率
    q: (k, vocab) 草稿模型抽取每个 token 时实际用的处理后概率
    draft_tokens: (k,)
    返回 (接受的草稿数, 本轮最后追加的 token,形状 (1, 1))
    '''
    k = draft_tokens.shape[0]
    for i in range(k):
        token = draft_tokens[i]
        r = torch.rand(1, device=p.device)
        # 以 min(1, p/q) 的概率接受;写成 >= 保证 p=0 时必拒绝
        if r >= p[i, token] / q[i, token]:
            return i, sample(max_fn(p[i : i + 1] - q[i : i + 1]))
    return k, sample(p[k : k + 1])


def speculative_sampling(
    prefix: torch.Tensor,
    approx_model: torch.nn.Module,
    target_model: torch.nn.Module,
    max_len: int,
    gamma: int = 4,
    temperature: float = 1,
    top_k: int = 0,
    top_p: float = 0,
) -> torch.Tensor:
    '''
    草稿模型一次猜 gamma 个 token,目标模型一次前向验证并接受或拒绝
    每轮开始时目标模型的 cache 覆盖 prefix[:-1],本轮只需喂入最后一个已确定 token 和 k 个草稿
    '''
    validate_sampling_args(temperature, top_p, gamma)
    target_len = prefix.shape[1] + max_len
    assert prefix.shape[0] == 1, 'input batch size must be 1'
    assert prefix.shape[1] > 0, 'prompt must contain at least one token'

    draft = KVCacheModel(approx_model)
    target = KVCacheModel(target_model)

    with torch.no_grad(), tqdm(total=max_len, desc='speculative sampling') as pbar:
        while prefix.shape[1] < target_len:
            prefix_len = prefix.shape[1]
            # 每轮最多产出 k 个草稿加 1 个修正/bonus token,k 不能让长度超过 target_len
            k = min(gamma, target_len - prefix_len - 1)

            x = prefix
            q_steps = []
            for _ in range(k):
                q_logits = draft(x, logits_to_keep=1)[:, -1, :]
                q_step = norm_logits(q_logits, temperature, top_k, top_p)
                q_steps.append(q_step)
                x = torch.cat((x, sample(q_step)), dim=1)

            target_logits = target(x, logits_to_keep=k + 1)[0]
            p = norm_logits(target_logits, temperature, top_k, top_p)
            q = torch.cat(q_steps, dim=0) if q_steps else p[:0]

            n_accepted, next_token = verify_draft(p, q, x[0, prefix_len:])
            # 被拒绝的草稿已经写进两边的 cache,回滚到最后一个被接受的位置
            draft.rollback(prefix_len + n_accepted)
            target.rollback(prefix_len + n_accepted)
            prefix = torch.cat((x[:, : prefix_len + n_accepted], next_token), dim=1)
            pbar.update(n_accepted + 1)

    return prefix


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    # Qwen2.5 权重本身是 bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return model


def generate(
    input_text: str,
    approx_model_name: str,
    target_model_name: str,
    num_tokens: int = 20,
    gamma: int = 4,
    temperature: float = 1.0,
    top_k: int = 10,
    top_p: float = 0.0,
    seed: int = 123,
) -> None:
    # 加载模型要十几秒，参数错误应在此之前报出
    validate_sampling_args(temperature, top_p, gamma)
    device = pick_device()
    print(f'device: {device}, dtype: bfloat16')

    tokenizer = AutoTokenizer.from_pretrained(approx_model_name, local_files_only=True)
    approx_model = load_model(approx_model_name, device)
    target_model = load_model(target_model_name, device)
    if approx_model.config.vocab_size != target_model.config.vocab_size:
        raise ValueError(
            'draft and target vocab sizes differ: '
            f'{approx_model.config.vocab_size} vs {target_model.config.vocab_size}'
        )

    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
    sampling = dict(temperature=temperature, top_k=top_k, top_p=top_p)

    def run_speculative(n: int) -> torch.Tensor:
        return speculative_sampling(input_ids, approx_model, target_model, n, gamma=gamma, **sampling)

    def run_autoregressive(n: int) -> torch.Tensor:
        return autoregressive_sampling(input_ids, target_model, n, **sampling)

    # MPS 第一次运行会编译 Metal kernel,计时前两条路径各预热一次
    run_speculative(gamma + 1)
    run_autoregressive(2)

    for name, run in (
        ('speculative_sampling', run_speculative),
        ('autoregressive_sampling', run_autoregressive),
    ):
        set_seed(seed)
        output_ids, elapsed = timed(lambda run=run: run(num_tokens), device)
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f'{name} ({elapsed:.2f}s, {num_tokens / elapsed:.1f} tok/s): {text}')


if __name__ == '__main__':
    args = parse_arguments()
    generate(
        args.input,
        args.approx_model_name,
        args.target_model_name,
        num_tokens=args.num_tokens,
        gamma=args.gamma,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
    )
