import contextlib
from types import SimpleNamespace

import pytest

import main


@pytest.fixture(autouse=True)
def no_progress_bar(monkeypatch):
    # 分布测试要跑上万次采样，tqdm 的创建与刷新开销会主导耗时
    monkeypatch.setattr(
        main,
        'tqdm',
        lambda *args, **kwargs: contextlib.nullcontext(SimpleNamespace(update=lambda n: None)),
    )
