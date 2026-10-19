import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def autoregressive_sampling(
    x: torch.Tensor,
    model: torch.nn.Module,
    n_new_tokens: int,
    temperature: float = 1,
    top_k: int = 0,
    top_p: float = 0,
) -> torch.Tensor:
    '''只用 target model 逐 token 生成,作为投机采样的对照。'''
    generated = 0
    with torch.no_grad(), tqdm(total=n_new_tokens, desc='autoregressive sampling') as pbar:
        while generated < n_new_tokens:
            logits = model(x, use_cache=False).logits[:, -1, :]
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
    '''草稿模型一次猜 gamma 个 token,目标模型一次前向验证并接受或拒绝。'''
    target_len = prefix.shape[1] + max_len
    assert prefix.shape[0] == 1, 'input batch size must be 1'
    assert prefix.shape[1] > 0, 'prompt must contain at least one token'

    with torch.no_grad(), tqdm(total=max_len, desc='speculative sampling') as pbar:
        while prefix.shape[1] < target_len:
            prefix_len = prefix.shape[1]
            # 每轮最多产出 k 个草稿加 1 个修正/bonus token,k 不能让长度超过 target_len
            k = min(gamma, target_len - prefix_len - 1)

            x = prefix
            q_steps = []
            for _ in range(k):
                q_logits = approx_model(x, use_cache=False).logits[:, -1, :]
                q_step = norm_logits(q_logits, temperature, top_k, top_p)
                q_steps.append(q_step)
                x = torch.cat((x, sample(q_step)), dim=1)

            target_logits = target_model(x, use_cache=False).logits[0, prefix_len - 1 :, :]
            p = norm_logits(target_logits, temperature, top_k, top_p)
            q = torch.cat(q_steps, dim=0) if q_steps else p[:0]

            n_accepted, next_token = verify_draft(p, q, x[0, prefix_len:])
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

    set_seed(seed)
    speculative_ids = speculative_sampling(
        input_ids,
        approx_model,
        target_model,
        num_tokens,
        gamma=gamma,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    speculative_text = tokenizer.decode(speculative_ids[0], skip_special_tokens=True)
    print(f'speculative_sampling: {speculative_text}')

    set_seed(seed)
    autoregressive_ids = autoregressive_sampling(
        input_ids,
        target_model,
        num_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    autoregressive_text = tokenizer.decode(autoregressive_ids[0], skip_special_tokens=True)
    print(f'autoregressive_sampling: {autoregressive_text}')


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
