from __future__ import annotations

import multiprocessing
from collections.abc import Mapping
from dataclasses import dataclass
from multiprocessing.synchronize import Barrier
from pathlib import Path
from types import MappingProxyType
from typing import Final, NoReturn

from cli_harness.rate_limiter import ProviderConfig, RateBucket, RateLimiter

_RACERS: Final = 8
_ATTEMPTS_PER_RACER: Final = 100
_BURST: Final = 400
_NO_REFILL: Final[Mapping[RateBucket, ProviderConfig]] = MappingProxyType(
    {RateBucket.OPENAI: ProviderConfig(rate_per_sec=1e-9, burst=_BURST)}
)


@dataclass
class _FakeTime:
    now: float = 0.0

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_two_processes_on_one_host_share_a_single_budget_per_bucket(tmp_path: Path) -> None:
    fake_time: Final = _FakeTime()
    config: Final = {RateBucket.OPENAI: ProviderConfig(rate_per_sec=2.0, burst=1.0)}
    first_process: Final = RateLimiter(config, tmp_path, fake_time.clock, fake_time.sleep)
    second_process: Final = RateLimiter(config, tmp_path, fake_time.clock, fake_time.sleep)

    assert first_process.acquire(RateBucket.OPENAI) == 0.0
    assert 0.5 <= second_process.acquire(RateBucket.OPENAI) < 0.51
    assert second_process.acquire(RateBucket.BEDROCK_MANTLE) == 0.0


def test_processes_racing_for_one_bucket_never_get_more_than_its_burst(tmp_path: Path) -> None:
    grants: Final = tmp_path / "grants"

    _race(_RACERS, state_dir=tmp_path / "state", grants=grants)

    assert len(grants.read_bytes()) == _BURST


def _race(racers: int, *, state_dir: Path, grants: Path) -> None:
    context: Final = multiprocessing.get_context("spawn")
    start_line: Final = context.Barrier(racers)
    processes: Final = tuple(
        context.Process(target=_take_tokens_until_told_to_wait, args=(state_dir, grants, start_line))
        for _ in range(racers)
    )
    for process in processes:
        process.start()
    for process in processes:
        process.join()


def _take_tokens_until_told_to_wait(state_dir: Path, grants: Path, start_line: Barrier) -> None:
    limiter: Final = RateLimiter(_NO_REFILL, state_dir, _frozen_clock, _exit_instead_of_sleeping)
    with grants.open("ab", buffering=0) as log:
        _ = start_line.wait()
        for _ in range(_ATTEMPTS_PER_RACER):
            _ = limiter.acquire(RateBucket.OPENAI)
            _ = log.write(b".")


def _frozen_clock() -> float:
    return 0.0


def _exit_instead_of_sleeping(_seconds: float) -> NoReturn:
    raise SystemExit(0)
