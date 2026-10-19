'''无损性：投机采样产出的 token 必须服从目标模型处理后的分布 p'。'''

import pytest
import torch
from toy_models import ContextFreeModel, assert_samples_follow, tv_distance

import main

VOCAB_SIZE = 8
TARGET_LOGITS = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5])
# 草稿分布刻意和目标错开，top-k 时两者的支撑集也不同
DRAFT_LOGITS = torch.tensor([-1.0, 0.0, 2.0, 1.0, -0.5, 1.5, 0.5, -1.5])

SAMPLING_CONFIGS = {
    'plain': dict(temperature=1.0, top_k=0, top_p=0.0),
    'temperature_top_k': dict(temperature=0.7, top_k=5, top_p=0.0),
    'temperature_top_p': dict(temperature=1.3, top_k=0, top_p=0.8),
}


def processed(logits: torch.Tensor, rows: int, config: dict) -> torch.Tensor:
    return main.norm_logits(logits.repeat(rows, 1), **config)


@pytest.mark.parametrize('config_name', SAMPLING_CONFIGS)
def test_fixture_distributions_are_distinguishable(config_name):
    # 若 p' 与 q' 或原始 softmax 很接近，下面的分布测试就测不出错误
    config = SAMPLING_CONFIGS[config_name]
    p = processed(TARGET_LOGITS, 1, config)[0]
    q = processed(DRAFT_LOGITS, 1, config)[0]
    assert tv_distance(p, q) > 0.3
    if config_name != 'plain':
        assert tv_distance(p, torch.softmax(TARGET_LOGITS, dim=-1)) > 0.05


@pytest.mark.parametrize('gamma', [1, 3])
@pytest.mark.parametrize('config_name', SAMPLING_CONFIGS)
def test_verify_draft_first_token_follows_target(config_name, gamma):
    '''一步验证跑 1e5 次，本轮第一个产出的 token 必须服从 p'。'''
    config = SAMPLING_CONFIGS[config_name]
    n_trials = 100_000
    torch.manual_seed(0)

    p = processed(TARGET_LOGITS, gamma + 1, config)
    q = processed(DRAFT_LOGITS, gamma, config)
    drafts = torch.multinomial(q[0], n_trials * gamma, replacement=True).view(n_trials, gamma)

    first_tokens = torch.empty(n_trials, dtype=torch.long)
    for trial in range(n_trials):
        n_accepted, next_token = main.verify_draft(p, q, drafts[trial])
        first_tokens[trial] = drafts[trial, 0] if n_accepted > 0 else next_token.item()

    counts = torch.bincount(first_tokens, minlength=VOCAB_SIZE)
    assert_samples_follow(counts, p[0], tv_tol=0.01)


@pytest.mark.parametrize(
    ('config_name', 'gamma'),
    [('temperature_top_k', 2), ('temperature_top_p', 1)],
)
def test_speculative_sampling_sequences_follow_target(config_name, gamma):
    '''
    端到端：上下文无关的目标模型下，生成序列应是 p' 的独立同分布样本。
    逐位置检查边缘分布，并对前两个位置的联合分布做卡方检验。
    '''
    config = SAMPLING_CONFIGS[config_name]
    n_runs = 20_000
    n_new_tokens = 3
    torch.manual_seed(0)

    draft = ContextFreeModel(DRAFT_LOGITS)
    target = ContextFreeModel(TARGET_LOGITS)
    prompt = torch.zeros(1, 1, dtype=torch.long)

    outputs = torch.empty(n_runs, n_new_tokens, dtype=torch.long)
    for run in range(n_runs):
        out = main.speculative_sampling(
            prompt, draft, target, n_new_tokens, gamma=gamma, **config
        )
        outputs[run] = out[0, 1:]

    p = processed(TARGET_LOGITS, 1, config)[0]
    for position in range(n_new_tokens):
        counts = torch.bincount(outputs[:, position], minlength=VOCAB_SIZE)
        assert_samples_follow(counts, p, tv_tol=0.02)

    pair_index = outputs[:, 0] * VOCAB_SIZE + outputs[:, 1]
    pair_counts = torch.bincount(pair_index, minlength=VOCAB_SIZE**2)
    assert_samples_follow(pair_counts, torch.outer(p, p).flatten(), tv_tol=0.04)


def test_norm_logits_does_not_mutate_input():
    # top_k_top_p_filter 是原地操作，norm_logits 不能把调用方的 logits 改掉
    logits = TARGET_LOGITS.repeat(2, 1)
    before = logits.clone()
    main.norm_logits(logits, temperature=1.0, top_k=3, top_p=0.5)
    assert torch.equal(logits, before)


def test_norm_logits_greedy_is_one_hot():
    logits = torch.tensor([[0.1, 3.0, -1.0], [5.0, 4.0, 4.5]])
    probs = main.norm_logits(logits, temperature=0, top_k=0, top_p=0.0)
    assert torch.equal(probs, torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))


@pytest.mark.parametrize('config_name', SAMPLING_CONFIGS)
def test_norm_logits_rows_are_distributions(config_name):
    probs = processed(TARGET_LOGITS, 3, SAMPLING_CONFIGS[config_name])
    assert probs.dtype == torch.float32
    assert torch.allclose(probs.sum(dim=-1), torch.ones(3))
    assert (probs >= 0).all()
