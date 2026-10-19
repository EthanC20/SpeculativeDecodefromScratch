'''temperature=0 时，投机解码必须与目标模型的 greedy 输出逐 token 一致。'''

import copy

import pytest
import torch
from toy_models import BigramModel, CallCounter, greedy_reference
from transformers import AutoModelForCausalLM

import main

TINY_QWEN2 = 'onnx-internal-testing/tiny-random-Qwen2ForCausalLM'
GREEDY = dict(temperature=0, top_k=0, top_p=0.0)


@pytest.mark.parametrize('gamma', [1, 2, 4])
@pytest.mark.parametrize('n_new_tokens', [1, 3, 10, 25])
def test_bigram_greedy_matches_target(gamma, n_new_tokens):
    vocab_size = 16
    draft = BigramModel(vocab_size, seed=1)
    target = BigramModel(vocab_size, seed=2)
    prompt = torch.tensor([[3, 7, 1]])

    expected = greedy_reference(target, prompt, n_new_tokens)
    actual = main.speculative_sampling(prompt, draft, target, n_new_tokens, gamma=gamma, **GREEDY)
    assert torch.equal(actual, expected)


@pytest.fixture(scope='module')
def tiny_qwen2():
    try:
        model = AutoModelForCausalLM.from_pretrained(TINY_QWEN2, dtype=torch.float32)
    except OSError as error:
        pytest.skip(f'cannot load {TINY_QWEN2}: {error}')
    return model.eval()


def make_draft(target: torch.nn.Module, kind: str) -> torch.nn.Module:
    torch.manual_seed(0)
    if kind == 'independent':
        # 同结构、重新随机初始化：几乎每个位置都会被拒绝
        return AutoModelForCausalLM.from_config(target.config).eval()
    # 目标模型加小噪声：一部分草稿被接受，一部分被拒绝
    draft = copy.deepcopy(target)
    with torch.no_grad():
        for param in draft.parameters():
            param.add_(0.05 * param.std() * torch.randn_like(param))
    return draft.eval()


@pytest.mark.parametrize('draft_kind', ['independent', 'perturbed'])
@pytest.mark.parametrize('gamma', [1, 4])
def test_tiny_qwen2_greedy_matches_target(tiny_qwen2, draft_kind, gamma):
    n_new_tokens = 16
    draft = make_draft(tiny_qwen2, draft_kind)
    target = CallCounter(tiny_qwen2)
    generator = torch.Generator().manual_seed(0)
    prompt = torch.randint(0, tiny_qwen2.config.vocab_size, (1, 6), generator=generator)

    expected = greedy_reference(tiny_qwen2, prompt, n_new_tokens)
    actual = main.speculative_sampling(prompt, draft, target, n_new_tokens, gamma=gamma, **GREEDY)
    assert torch.equal(actual, expected)

    autoregressive = main.autoregressive_sampling(prompt, tiny_qwen2, n_new_tokens, **GREEDY)
    assert torch.equal(autoregressive, expected)

    if draft_kind == 'perturbed' and gamma > 1:
        # 至少有一轮接受了草稿，确保接受路径真的被走到
        assert target.calls < n_new_tokens
