"""A token bucket per upstream, shared by every process on the host, so parallel CLI suites and xdist workers
together stay under one request rate per provider instead of one rate each.

Each bucket is a small JSON file read and written under an exclusive `flock`, in `LITELLM_COMPAT_RATE_STATE_DIR`
(default: a fixed folder under `$TMPDIR`). `LITELLM_COMPAT_RATE_<BUCKET>` sets a bucket's req/s (default 5, 0 or less
turns it off) and `LITELLM_COMPAT_RATE_BURST` overrides every bucket's burst (default: its rate).
"""

from __future__ import annotations

import fcntl
import os
import tempfile
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, ValidationError

DEFAULT_RATE: Final = 5.0
RATE_ENV_PREFIX: Final = "LITELLM_COMPAT_RATE_"
BURST_ENV: Final = "LITELLM_COMPAT_RATE_BURST"
STATE_DIR_ENV: Final = "LITELLM_COMPAT_RATE_STATE_DIR"
DEFAULT_STATE_DIR_NAME: Final = "litellm-claude-compat-ratelimit"
_WAKE_MARGIN_SECONDS: Final = 1e-3


class RateBucket(StrEnum):
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    VERTEX_AI = "vertex_ai"
    BEDROCK_CONVERSE = "bedrock_converse"
    BEDROCK_INVOKE = "bedrock_invoke"
    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    BEDROCK_MANTLE = "bedrock_mantle"


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    rate_per_sec: float
    burst: float

    @property
    def enabled(self) -> bool:
        return self.rate_per_sec > 0 and self.burst > 0


class _BucketState(BaseModel):
    model_config = ConfigDict(frozen=True)

    tokens: float
    last_refill: float


def load_config(env: Mapping[str, str] = os.environ) -> Mapping[RateBucket, ProviderConfig]:
    burst_override: Final = _as_float(env.get(BURST_ENV), -1.0)
    return MappingProxyType(
        {
            bucket: _bucket_config(_as_float(env.get(_rate_env(bucket)), DEFAULT_RATE), burst_override)
            for bucket in RateBucket
        }
    )


def default_state_dir(env: Mapping[str, str] = os.environ) -> Path:
    explicit: Final = env.get(STATE_DIR_ENV)
    return Path(explicit) if explicit else Path(tempfile.gettempdir()) / DEFAULT_STATE_DIR_NAME


@dataclass(frozen=True, slots=True)
class RateLimiter:
    config: Mapping[RateBucket, ProviderConfig] = field(default_factory=load_config)
    state_dir: Path = field(default_factory=default_state_dir)
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def acquire(self, bucket: RateBucket) -> float:
        cfg: Final = self.config.get(bucket)
        if cfg is None or not cfg.enabled:
            return 0.0
        return sum(self._waits(self.state_dir / f"{bucket.value}.json", cfg))

    def _waits(self, path: Path, cfg: ProviderConfig) -> Iterator[float]:
        while (wait := _try_consume(path, cfg, self.clock())) > 0:
            self.sleep(wait)
            yield wait


@cache
def get_default_limiter() -> RateLimiter:
    return RateLimiter()


def _rate_env(bucket: RateBucket) -> str:
    return RATE_ENV_PREFIX + bucket.value.upper()


def _bucket_config(rate: float, burst_override: float) -> ProviderConfig:
    return ProviderConfig(rate_per_sec=rate, burst=burst_override if burst_override > 0 else max(rate, 1.0))


def _as_float(value: str | None, default: float) -> float:
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _try_consume(path: Path, cfg: ProviderConfig, now: float) -> float:
    fd: Final = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        state: Final = _refilled(_read_state(fd, cfg, now), now, cfg)
        if state.tokens >= 1.0:
            _write_state(fd, _BucketState(tokens=state.tokens - 1.0, last_refill=state.last_refill))
            return 0.0
        _write_state(fd, state)
        return (1.0 - state.tokens) / cfg.rate_per_sec + _WAKE_MARGIN_SECONDS
    finally:
        os.close(fd)


def _read_state(fd: int, cfg: ProviderConfig, now: float) -> _BucketState:
    os.lseek(fd, 0, os.SEEK_SET)
    raw: Final = os.read(fd, 4096)
    try:
        return _BucketState.model_validate_json(raw)
    except ValidationError:
        return _BucketState(tokens=cfg.burst, last_refill=now)


def _refilled(state: _BucketState, now: float, cfg: ProviderConfig) -> _BucketState:
    elapsed: Final = max(0.0, now - state.last_refill)
    return _BucketState(tokens=min(cfg.burst, state.tokens + elapsed * cfg.rate_per_sec), last_refill=now)


def _write_state(fd: int, state: _BucketState) -> None:
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    os.write(fd, state.model_dump_json().encode("utf-8"))
