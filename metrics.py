'''解码耗时与投机采样接受率统计。'''

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from contexttimer import Timer


@dataclass
class SpeculativeStats:
    '''按轮累加：每轮目标模型前向一次，验证 drafted 个草稿并接受其中 accepted 个。'''

    rounds: int = 0
    drafted: int = 0
    accepted: int = 0

    def record(self, drafted: int, accepted: int) -> None:
        self.rounds += 1
        self.drafted += drafted
        self.accepted += accepted

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def accepted_per_round(self) -> float:
        return self.accepted / self.rounds if self.rounds else 0.0

    @property
    def tokens_per_round(self) -> float:
        # 每轮在接受的草稿之外，还会追加 1 个修正或 bonus token
        return (self.accepted + self.rounds) / self.rounds if self.rounds else 0.0

    def summary(self) -> str:
        return (
            f'{self.rounds} rounds, accepted {self.accepted}/{self.drafted} drafts '
            f'({self.acceptance_rate:.1%}), {self.accepted_per_round:.2f} accepted and '
            f'{self.tokens_per_round:.2f} tokens per target forward'
        )


def synchronize(device: torch.device) -> None:
    # MPS 是异步提交，不同步的话测到的只是提交时间
    if device.type == 'mps':
        torch.mps.synchronize()


@contextmanager
def device_timer(device: torch.device) -> Iterator[Timer]:
    '''contexttimer.Timer,进入前和退出前各同步一次设备。'''
    synchronize(device)
    with Timer() as timer:
        yield timer
        synchronize(device)


def throughput_summary(n_tokens: int, elapsed: float) -> str:
    return f'{n_tokens} tokens in {elapsed:.2f}s ({n_tokens / elapsed:.1f} tok/s)'
