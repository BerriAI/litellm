from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from cli_harness.rate_limiter import ProviderConfig, RateBucket, RateLimiter


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
