'''KV cache 与 logits_to_keep:结果必须与整段重算一致，且每个位置只算一次。'''

import math

import pytest
import torch
from toy_models import (
    ContextFreeModel,
    PrefixSumModel,
    greedy_reference,
    greedy_speculative_reference,
)
from transformers import AutoModelForCausalLM

import main

VOCAB_SIZE = 16
PROMPT = torch.tensor([[3, 7, 1, 9]])
GREEDY = dict(temperature=0, top_k=0, top_p=0.0)
SAMPLING = dict(temperature=1.0, top_k=0, top_p=0.0)
TINY_QWEN2 = 'onnx-internal-testing/tiny-random-Qwen2ForCausalLM'

# (追加的 token, logits_to_keep, 追加前回滚到的长度);None 表示不回滚
CACHE_SCRIPT = [
    (None, 1, None),
    ([5, 2, 8], 3, None),
    ([11], 1, 5),
    ([4, 4], 2, None),
    ([1, 6, 0], 3, 4),
]


def perturbed_pair(seed: int) -> tuple[PrefixSumModel, PrefixSumModel]:
    '''目标模型和一个加了噪声的草稿：greedy 下既有接受也有拒绝。'''
    target = PrefixSumModel(VOCAB_SIZE, seed=seed)
    draft = PrefixSumModel(VOCAB_SIZE, seed=seed)
    generator = torch.Generator().manual_seed(seed + 1000)
    draft.table += 2.0 * torch.randn(draft.table.shape, generator=generator)
    return draft, target


def replay_cache_script(model: torch.nn.Module, full_forward) -> None:
    '''按 CACHE_SCRIPT 增量调用 KVCacheModel,每一步都与整段重算比较。'''
    cached = main.KVCacheModel(model)
    x = PROMPT
    with torch.no_grad():
        for new_tokens, keep, rollback_to in CACHE_SCRIPT:
            if rollback_to is not None:
                cached.rollback(rollback_to)
                x = x[:, :rollback_to]
            if new_tokens is not None:
                x = torch.cat((x, torch.tensor([new_tokens])), dim=1)
            actual = cached(x, logits_to_keep=keep)
            expected = full_forward(x)[:, -keep:, :]
            assert actual.shape == expected.shape
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
            assert cached.cache.get_seq_length() == x.shape[1]


def test_kv_cache_model_matches_full_recompute_on_toy():
    model = PrefixSumModel(VOCAB_SIZE, seed=0)
    replay_cache_script(model, lambda x: model(x).logits)


@pytest.fixture(scope='module')
def tiny_qwen2():
    try:
        model = AutoModelForCausalLM.from_pretrained(TINY_QWEN2, dtype=torch.float32)
    except OSError as error:
        pytest.skip(f'cannot load {TINY_QWEN2}: {error}')
    return model.eval()


def test_kv_cache_model_matches_full_recompute_on_tiny_qwen2(tiny_qwen2):
    replay_cache_script(tiny_qwen2, lambda x: tiny_qwen2(x, use_cache=False).logits)


@pytest.mark.parametrize('gamma', [1, 2, 4])
@pytest.mark.parametrize('n_new_tokens', [1, 5, 20])
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_rollback_keeps_greedy_rounds_identical(seed, gamma, n_new_tokens):
    '''
    logits 依赖整个前缀：回滚后任一 cache 残留被拒绝的草稿，
    输出或接受/拒绝的轮次就会和不带 cache 的参考实现不同
    '''
    draft, target = perturbed_pair(seed)
    expected, target_calls, draft_calls = greedy_speculative_reference(
        draft, target, PROMPT, n_new_tokens, gamma
    )
    assert torch.equal(expected, greedy_reference(target, PROMPT, n_new_tokens))
    draft.reset_stats()
    target.reset_stats()

    actual = main.speculative_sampling(PROMPT, draft, target, n_new_tokens, gamma=gamma, **GREEDY)

    assert torch.equal(actual, expected)
    assert target.calls == target_calls
    assert draft.calls == draft_calls


def test_perturbed_pair_exercises_accept_and_reject():
    # 确保上面的参数组合里真的同时出现了接受和拒绝
    draft, target = perturbed_pair(0)
    n_new_tokens, gamma = 20, 4
    _, target_calls, _ = greedy_speculative_reference(draft, target, PROMPT, n_new_tokens, gamma)
    assert math.ceil(n_new_tokens / (gamma + 1)) < target_calls < n_new_tokens


@pytest.mark.parametrize('gamma', [1, 3])
@pytest.mark.parametrize('n_new_tokens', [1, 4, 11])
def test_lm_head_rows_and_tokens_fed_when_all_accepted(gamma, n_new_tokens):
    '''
    草稿与目标相同 → 全部接受,cache 从不回滚：
    每个位置恰好喂入目标模型一次，草稿每步只算 1 行 logits,目标每轮只算 k+1 行
    '''
    logits = torch.tensor([0.3, -1.0, 2.0, 0.0, 1.0, -0.5])
    draft = ContextFreeModel(logits)
    target = ContextFreeModel(logits)
    prompt = torch.tensor([[2, 4, 1]])

    torch.manual_seed(0)
    main.speculative_sampling(prompt, draft, target, n_new_tokens, gamma=gamma, **SAMPLING)

    remaining, expected_rows = n_new_tokens, []
    while remaining > 0:
        expected_rows.append(min(gamma + 1, remaining))
        remaining -= expected_rows[-1]
    assert target.logit_rows == expected_rows
    assert set(draft.logit_rows) <= {1}
    assert target.tokens_fed == prompt.shape[1] + n_new_tokens - 1
    assert draft.tokens_fed <= prompt.shape[1] + n_new_tokens - 1


@pytest.mark.parametrize('gamma', [1, 3])
def test_tokens_fed_when_every_draft_is_rejected(gamma):
    '''每轮在第一个草稿就被拒绝：目标模型每轮只多喂 k+1 个 token,而不是整段前缀。'''
    n_new_tokens = 9
    draft_logits = torch.full((6,), float('-inf'))
    draft_logits[0] = 0.0
    target_logits = torch.zeros(6)
    target_logits[0] = float('-inf')
    draft = ContextFreeModel(draft_logits)
    target = ContextFreeModel(target_logits)
    prompt = torch.tensor([[2, 4, 1]])

    torch.manual_seed(0)
    main.speculative_sampling(prompt, draft, target, n_new_tokens, gamma=gamma, **SAMPLING)

    drafts_per_round = [min(gamma, n_new_tokens - i - 1) for i in range(n_new_tokens)]
    assert target.logit_rows == [k + 1 for k in drafts_per_round]
    assert target.tokens_fed == prompt.shape[1] - 1 + sum(k + 1 for k in drafts_per_round)
    assert set(draft.logit_rows) <= {1}


def test_autoregressive_uses_cache_and_single_logit_row():
    model = PrefixSumModel(VOCAB_SIZE, seed=3)
    n_new_tokens = 12
    expected = greedy_reference(model, PROMPT, n_new_tokens)
    model.reset_stats()

    actual = main.autoregressive_sampling(PROMPT, model, n_new_tokens, **GREEDY)

    assert torch.equal(actual, expected)
    assert model.logit_rows == [1] * n_new_tokens
    assert model.tokens_fed == PROMPT.shape[1] + n_new_tokens - 1
