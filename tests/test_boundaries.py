'''边界情况：γ=1、首位即拒绝、全部接受、num_tokens < γ，以及输出长度。'''

import math

import pytest
import torch
from toy_models import BigramModel, ContextFreeModel

import main

VOCAB_SIZE = 6
PROMPT = torch.tensor([[2, 4]])
SAMPLING = dict(temperature=1.0, top_k=0, top_p=0.0)


def rounds_and_drafts(n_new_tokens: int, gamma: int) -> tuple[int, int]:
    '''全部接受时的目标模型调用次数与草稿模型调用次数。'''
    target_calls = math.ceil(n_new_tokens / (gamma + 1))
    return target_calls, n_new_tokens - target_calls


@pytest.mark.parametrize('gamma', [1, 2, 3, 5])
@pytest.mark.parametrize('n_new_tokens', list(range(13)))
def test_output_length_is_exact(gamma, n_new_tokens):
    torch.manual_seed(gamma * 100 + n_new_tokens)
    draft = BigramModel(VOCAB_SIZE, seed=1)
    target = BigramModel(VOCAB_SIZE, seed=2)
    out = main.speculative_sampling(PROMPT, draft, target, n_new_tokens, gamma=gamma, **SAMPLING)
    assert out.shape == (1, PROMPT.shape[1] + n_new_tokens)
    assert torch.equal(out[:, : PROMPT.shape[1]], PROMPT)


@pytest.mark.parametrize('gamma', [1, 3])
def test_rejection_at_first_position(gamma):
    '''草稿总是猜 token 0，而 p'(0)=0：每轮在第一个位置就被拒绝。'''
    n_new_tokens = 9
    draft_logits = torch.full((VOCAB_SIZE,), float('-inf'))
    draft_logits[0] = 0.0
    target_logits = torch.zeros(VOCAB_SIZE)
    target_logits[0] = float('-inf')
    draft = ContextFreeModel(draft_logits)
    target = ContextFreeModel(target_logits)

    torch.manual_seed(0)
    out = main.speculative_sampling(PROMPT, draft, target, n_new_tokens, gamma=gamma, **SAMPLING)
    generated = out[0, PROMPT.shape[1] :]

    assert generated.numel() == n_new_tokens
    assert (generated != 0).all()
    # 每轮只产出一个修正 token
    assert target.calls == n_new_tokens


def test_residual_after_rejection_follows_target():
    '''拒绝后的修正 token 来自 norm(max(0, p' - q'))；这里 q' 是 one-hot，残差应正好等于 p'。'''
    draft_logits = torch.full((VOCAB_SIZE,), float('-inf'))
    draft_logits[0] = 0.0
    target_logits = torch.tensor([float('-inf'), 1.0, 0.5, 0.0, -0.5, -1.0])
    p = main.norm_logits(target_logits.repeat(2, 1), **SAMPLING)
    q = main.norm_logits(draft_logits.repeat(1, 1), **SAMPLING)

    torch.manual_seed(0)
    tokens = []
    for _ in range(20_000):
        n_accepted, next_token = main.verify_draft(p, q, torch.tensor([0]))
        assert n_accepted == 0
        tokens.append(next_token.item())
    freq = torch.bincount(torch.tensor(tokens), minlength=VOCAB_SIZE).double() / len(tokens)
    assert freq[0] == 0
    assert 0.5 * (freq - p[0].double()).abs().sum() < 0.02


@pytest.mark.parametrize('gamma', [1, 2, 4])
@pytest.mark.parametrize('n_new_tokens', [1, 5, 10])
def test_all_drafts_accepted_when_models_match(gamma, n_new_tokens):
    '''草稿与目标完全相同时 p'/q'=1，每个草稿都应被接受。'''
    logits = torch.tensor([0.3, -1.0, 2.0, 0.0, 1.0, -0.5])
    draft = ContextFreeModel(logits)
    target = ContextFreeModel(logits)

    torch.manual_seed(0)
    out = main.speculative_sampling(PROMPT, draft, target, n_new_tokens, gamma=gamma, **SAMPLING)

    assert out.shape[1] == PROMPT.shape[1] + n_new_tokens
    expected_target_calls, expected_draft_calls = rounds_and_drafts(n_new_tokens, gamma)
    assert target.calls == expected_target_calls
    assert draft.calls == expected_draft_calls


@pytest.mark.parametrize(('n_new_tokens', 'gamma'), [(0, 4), (1, 4), (2, 4), (3, 4)])
def test_fewer_tokens_than_gamma(n_new_tokens, gamma):
    '''num_tokens <= γ 时只跑一轮，草稿数被截到 num_tokens - 1。'''
    logits = torch.tensor([0.3, -1.0, 2.0, 0.0, 1.0, -0.5])
    draft = ContextFreeModel(logits)
    target = ContextFreeModel(logits)

    torch.manual_seed(0)
    out = main.speculative_sampling(PROMPT, draft, target, n_new_tokens, gamma=gamma, **SAMPLING)

    assert out.shape[1] == PROMPT.shape[1] + n_new_tokens
    assert target.calls == (1 if n_new_tokens > 0 else 0)
    assert draft.calls == max(n_new_tokens - 1, 0)
