'''SpeculativeStats:轮数、草稿数、接受数与接受率。'''

import math

import pytest
import torch
from toy_models import BigramModel, ContextFreeModel, PrefixSumModel, greedy_speculative_reference

import main

PROMPT = torch.tensor([[2, 4, 1]])
SAMPLING = dict(temperature=1.0, top_k=0, top_p=0.0)
GREEDY = dict(temperature=0, top_k=0, top_p=0.0)


def run_with_stats(draft, target, n_new_tokens, gamma, **sampling):
    stats = main.SpeculativeStats()
    out = main.speculative_sampling(
        PROMPT, draft, target, n_new_tokens, gamma=gamma, stats=stats, **sampling
    )
    return out, stats


@pytest.mark.parametrize('gamma', [1, 2, 4])
@pytest.mark.parametrize('n_new_tokens', [0, 1, 7, 20])
def test_each_round_yields_accepted_plus_one_token(gamma, n_new_tokens):
    torch.manual_seed(gamma * 100 + n_new_tokens)
    draft = BigramModel(8, seed=1)
    target = BigramModel(8, seed=2)
    _, stats = run_with_stats(draft, target, n_new_tokens, gamma, **SAMPLING)

    assert stats.accepted + stats.rounds == n_new_tokens
    assert stats.rounds == target.calls
    assert stats.drafted == draft.calls
    assert 0 <= stats.accepted <= stats.drafted <= gamma * stats.rounds


@pytest.mark.parametrize('gamma', [1, 3])
def test_identical_models_accept_everything(gamma):
    n_new_tokens = 10
    logits = torch.tensor([0.3, -1.0, 2.0, 0.0, 1.0, -0.5])
    torch.manual_seed(0)
    _, stats = run_with_stats(
        ContextFreeModel(logits), ContextFreeModel(logits), n_new_tokens, gamma, **SAMPLING
    )

    assert stats.rounds == math.ceil(n_new_tokens / (gamma + 1))
    assert stats.accepted == stats.drafted == n_new_tokens - stats.rounds
    assert stats.acceptance_rate == 1.0


@pytest.mark.parametrize('gamma', [1, 3])
def test_disjoint_support_accepts_nothing(gamma):
    n_new_tokens = 9
    draft_logits = torch.full((6,), float('-inf'))
    draft_logits[0] = 0.0
    target_logits = torch.zeros(6)
    target_logits[0] = float('-inf')
    torch.manual_seed(0)
    _, stats = run_with_stats(
        ContextFreeModel(draft_logits), ContextFreeModel(target_logits), n_new_tokens, gamma,
        **SAMPLING,
    )

    assert stats.rounds == n_new_tokens
    assert stats.accepted == 0
    assert stats.drafted == sum(min(gamma, n_new_tokens - i - 1) for i in range(n_new_tokens))
    assert stats.acceptance_rate == 0.0
    assert stats.accepted_per_round == 0.0


@pytest.mark.parametrize('gamma', [2, 4])
def test_greedy_stats_match_reference(gamma):
    n_new_tokens = 20
    target = PrefixSumModel(16, seed=0)
    draft = PrefixSumModel(16, seed=0)
    generator = torch.Generator().manual_seed(1000)
    draft.table += 2.0 * torch.randn(draft.table.shape, generator=generator)

    _, target_calls, draft_calls = greedy_speculative_reference(
        draft, target, PROMPT, n_new_tokens, gamma
    )
    _, stats = run_with_stats(draft, target, n_new_tokens, gamma, **GREEDY)

    assert stats.rounds == target_calls
    assert stats.drafted == draft_calls
    assert stats.accepted == n_new_tokens - target_calls
    assert stats.accepted_per_round == pytest.approx(stats.accepted / stats.rounds)


def test_empty_stats_are_zero():
    stats = main.SpeculativeStats()
    assert stats.acceptance_rate == 0.0
    assert stats.accepted_per_round == 0.0
