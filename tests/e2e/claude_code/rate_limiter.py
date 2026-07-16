"""Cross-process token-bucket rate limiter for the Claude Code compat suite.

The compat matrix runs 75 live `claude` CLI invocations (25 cells × 3
Claude tiers per cell). When pytest-xdist fans these out across worker
processes, each worker would maintain its own in-memory rate limiter
and the *aggregate* request rate hitting any one upstream provider
would be `workers × per-worker rate` — exactly the situation that
trips Anthropic / Azure / Bedrock / Vertex 429s in the middle of a
matrix run and silently flips green cells red.

The fix is a **shared, cross-process** token bucket per provider,
backed by a small JSON state file and an OS-level `flock`. Each
`run_claude` invocation acquires one token (sleeping if the bucket is
empty) before launching the CLI; refills happen lazily based on wall
time, so workers can be killed and restarted without losing or
double-spending budget.

Configuration is driven entirely by environment variables so a
binary-search workflow can shift per-provider rates without code
edits:

    LITELLM_COMPAT_RATE_ANTHROPIC          (req/s, default 5.0)
    LITELLM_COMPAT_RATE_AZURE              (req/s, default 5.0)
    LITELLM_COMPAT_RATE_VERTEX_AI          (req/s, default 5.0)
    LITELLM_COMPAT_RATE_BEDROCK_CONVERSE   (req/s, default 5.0)
    LITELLM_COMPAT_RATE_BEDROCK_INVOKE     (req/s, default 5.0)
    LITELLM_COMPAT_RATE_BURST              (per-bucket burst override;
                                            default = rate)
    LITELLM_COMPAT_RATE_STATE_DIR          (state file directory;
                                            default = $TMPDIR/litellm-claude-compat-ratelimit)

A rate of 0 (or any non-positive value) disables throttling for that
provider — useful when you trust the upstream to handle the burst or
when running unit-test-shaped workloads that never actually hit the
network.

The provider id is inferred from the model id by `infer_provider`,
mirroring the matrix's column layout (`anthropic`, `azure`,
`vertex_ai`, `bedrock_converse`, `bedrock_invoke`).
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Generator, Mapping, Optional

from pydantic import ValidationError

from claude_code.json_types import JSON_OBJECT_ADAPTER, JSONValue

# `fcntl` is POSIX-only; the suite is Linux/macOS only, so we don't
# attempt a Windows fallback. Importing at module load fails fast on
# the (currently non-existent) Windows runner so we don't silently
# degrade to no-locking behavior.
import fcntl

PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_AZURE = "azure"
PROVIDER_VERTEX_AI = "vertex_ai"
PROVIDER_BEDROCK_CONVERSE = "bedrock_converse"
PROVIDER_BEDROCK_INVOKE = "bedrock_invoke"

ALL_PROVIDERS = (
    PROVIDER_ANTHROPIC,
    PROVIDER_AZURE,
    PROVIDER_VERTEX_AI,
    PROVIDER_BEDROCK_CONVERSE,
    PROVIDER_BEDROCK_INVOKE,
)

DEFAULT_RATE = 5.0  # req/s per provider, conservative starting point
RATE_ENV_PREFIX = "LITELLM_COMPAT_RATE_"
BURST_ENV = "LITELLM_COMPAT_RATE_BURST"
STATE_DIR_ENV = "LITELLM_COMPAT_RATE_STATE_DIR"
DEFAULT_STATE_DIR_NAME = "litellm-claude-compat-ratelimit"


def infer_provider(model: str) -> str:
    """Map a model alias to its compat-matrix provider id.

    The matrix column layout is fixed; aliases registered in the proxy
    encode the provider via a suffix (`-bedrock-converse`,
    `-bedrock-invoke`, `-azure`, `-vertex`) or its absence (Anthropic).
    Order matters: the bedrock suffixes both contain `bedrock`, so we
    test the more-specific ones first.
    """
    if not model:
        raise ValueError("model must be a non-empty string")
    lower = model.lower()
    if lower.endswith("-bedrock-converse"):
        return PROVIDER_BEDROCK_CONVERSE
    if lower.endswith("-bedrock-invoke"):
        return PROVIDER_BEDROCK_INVOKE
    if lower.endswith("-azure"):
        return PROVIDER_AZURE
    if lower.endswith("-vertex"):
        return PROVIDER_VERTEX_AI
    return PROVIDER_ANTHROPIC


@dataclass(frozen=True)
class ProviderConfig:
    """Static config snapshot for one provider's bucket.

    Captured up front (rather than re-read per acquire) so the limiter's
    behavior in a single process is stable even if env vars are mutated
    mid-run. A fresh `RateLimiter` picks up env changes on construction.
    """

    rate_per_sec: float
    burst: float

    @property
    def enabled(self) -> bool:
        return self.rate_per_sec > 0 and self.burst > 0


def load_config(
    env: Optional[Mapping[str, str]] = None,
) -> Dict[str, ProviderConfig]:
    """Build a {provider: ProviderConfig} from env, applying defaults.

    Parsing failures fall back to the default rate rather than
    crashing the test session — a typo in `LITELLM_COMPAT_RATE_AZURE`
    should not silently disable throttling, but it also shouldn't
    abort 75 live tests with a `ValueError` ten minutes in.
    """
    src = env if env is not None else os.environ

    def _as_float(value: Optional[str], default: float) -> float:
        if value is None or value == "":
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    burst_override = _as_float(src.get(BURST_ENV), -1.0)

    out: Dict[str, ProviderConfig] = {}
    for provider in ALL_PROVIDERS:
        env_key = RATE_ENV_PREFIX + provider.upper()
        rate = _as_float(src.get(env_key), DEFAULT_RATE)
        burst = burst_override if burst_override > 0 else max(rate, 1.0)
        out[provider] = ProviderConfig(rate_per_sec=rate, burst=burst)
    return out


def _state_dir(env: Optional[Mapping[str, str]] = None) -> Path:
    """Resolve the directory holding per-provider state files.

    A user-supplied `LITELLM_COMPAT_RATE_STATE_DIR` wins for tests and
    container environments where `$TMPDIR` may be ephemeral or shared
    in surprising ways. The directory is created lazily with
    `parents=True, exist_ok=True` so first-run setup needs no
    fixture wiring.
    """
    src = env if env is not None else os.environ
    explicit = src.get(STATE_DIR_ENV)
    if explicit:
        return Path(explicit)
    return Path(tempfile.gettempdir()) / DEFAULT_STATE_DIR_NAME


def _state_float(value: JSONValue) -> float:
    """Convert a JSON state value the way `float(...)` would, for the
    caller's corrupt-state handling: numeric and numeric-string values
    convert, everything else raises into the caller's except clause."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"cannot convert {type(value).__name__} to float")


class RateLimiter:
    """Cross-process token bucket per provider.

    Each `acquire(provider)` call:
      1. Opens (creating if needed) `<state_dir>/<provider>.json`.
      2. Holds an exclusive `flock` while reading + updating the
         {tokens, last_refill} state.
      3. Refills tokens based on `now - last_refill`, capped at burst.
      4. If tokens >= 1, subtracts one and returns immediately.
      5. Otherwise, computes the wall-time delay needed to earn a
         single token at the configured rate, releases the lock, and
         sleeps. After sleeping it retries — staying under the lock
         while sleeping would serialize all workers behind whichever
         one held it longest.

    This bounds the *aggregate* req/s seen by the upstream, regardless
    of how many xdist workers, threads, or processes are concurrently
    running tests against the same provider.

    A `_clock` / `_sleep` injection seam keeps the unit tests fast and
    deterministic; production callers should never override either.
    """

    def __init__(
        self,
        config: Optional[Mapping[str, ProviderConfig]] = None,
        state_dir: Optional[Path] = None,
        clock: Optional[Callable[[], float]] = None,
        sleep: Optional[Callable[[float], None]] = None,
    ) -> None:
        self._config = dict(config) if config is not None else load_config()
        self._state_dir = Path(state_dir) if state_dir is not None else _state_dir()
        self._clock = clock or time.monotonic
        self._sleep = sleep or time.sleep
        # `_state_dir.mkdir` once on construction is fine; concurrent
        # workers all racing to create the same directory is benign.
        self._state_dir.mkdir(parents=True, exist_ok=True)

    def acquire(self, provider: str) -> float:
        """Block until one token is available for `provider`.

        Returns the cumulative wall-time spent waiting (0.0 when the
        bucket had budget and we returned immediately). Callers can
        log this to attribute slow cells to throttling vs. upstream
        latency — same role `DriverResult.duration_ms` plays for the
        actual CLI invocation.
        """
        cfg = self._config.get(provider)
        if cfg is None or not cfg.enabled:
            return 0.0

        path = self._state_path(provider)
        total_waited = 0.0
        while True:
            now = self._clock()
            sleep_for = self._try_consume(path, cfg, now)
            if sleep_for <= 0:
                return total_waited
            self._sleep(sleep_for)
            total_waited += sleep_for

    def _state_path(self, provider: str) -> Path:
        return self._state_dir / f"{provider}.json"

    def _try_consume(self, path: Path, cfg: ProviderConfig, now: float) -> float:
        """Atomically refill and try to take one token.

        Returns 0.0 if a token was consumed, or a positive sleep
        duration (seconds) if the caller must wait before retrying.

        We hold an exclusive `flock` only across the read-modify-write
        of the JSON state — never across a `sleep` — so workers don't
        serialize while one of them is parked.
        """
        # `os.open` + `os.O_CREAT | os.O_RDWR` gives us a fd we can
        # both lock and read/write through. Opening with `"a+"` then
        # seeking is equivalent but uglier; this version is closer to
        # the canonical flock recipe.
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                tokens, last_refill = self._read_state(fd, cfg, now)
                tokens, last_refill = self._refill(tokens, last_refill, now, cfg)
                if tokens >= 1.0:
                    tokens -= 1.0
                    self._write_state(fd, tokens, last_refill)
                    return 0.0
                # Not enough budget. Persist the refilled state so a
                # subsequent caller doesn't have to redo the math, then
                # release the lock and tell the caller how long to
                # sleep before retrying.
                self._write_state(fd, tokens, last_refill)
                deficit = 1.0 - tokens
                # `deficit / rate` seconds will earn exactly enough
                # for one token. Add a tiny safety margin so we don't
                # wake up nanoseconds early and spin.
                return (deficit / cfg.rate_per_sec) + 1e-3
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    @staticmethod
    def _read_state(fd: int, cfg: ProviderConfig, now: float) -> tuple[float, float]:
        """Read {tokens, last_refill} from `fd`, defaulting to a full
        bucket on a missing/empty/corrupt file.

        New files start full so the very first request never waits;
        corrupt files are treated like new files because the
        alternative — refusing to run — is worse than briefly
        over-spending one bucket's worth of budget.
        """
        os.lseek(fd, 0, os.SEEK_SET)
        raw = os.read(fd, 4096).decode("utf-8")
        if not raw.strip():
            return cfg.burst, now
        try:
            obj = JSON_OBJECT_ADAPTER.validate_json(raw)
            tokens = _state_float(obj.get("tokens", cfg.burst))
            last_refill = _state_float(obj.get("last_refill", now))
            return tokens, last_refill
        except (ValidationError, ValueError, TypeError):
            return cfg.burst, now

    @staticmethod
    def _refill(
        tokens: float, last_refill: float, now: float, cfg: ProviderConfig
    ) -> tuple[float, float]:
        """Apply elapsed time to the bucket, capped at burst.

        Negative elapsed (clock went backward, e.g. across host
        sleep/resume or a manually-tweaked monotonic mock) is clamped
        to zero so we never *remove* tokens.
        """
        elapsed = max(0.0, now - last_refill)
        tokens = min(cfg.burst, tokens + elapsed * cfg.rate_per_sec)
        return tokens, now

    @staticmethod
    def _write_state(fd: int, tokens: float, last_refill: float) -> None:
        payload = json.dumps({"tokens": tokens, "last_refill": last_refill}).encode(
            "utf-8"
        )
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, payload)


# A single process-wide instance is all we need: provider-keyed state
# is stored in files, so multiple `RateLimiter` instances would just
# duplicate the in-process bookkeeping. We expose a getter rather than
# the instance directly so unit tests can install a custom limiter
# scoped to a tmp directory without monkeypatching globals.
_default: Optional[RateLimiter] = None


def get_default_limiter() -> RateLimiter:
    global _default
    if _default is None:
        _default = RateLimiter()
    return _default


def reset_default_limiter() -> None:
    """Drop the cached default limiter; the next `get_default_limiter`
    call rebuilds it from the current environment.

    Useful between unit tests that patch env vars: without this they'd
    keep reading the stale config snapshot from the first call.
    """
    global _default
    _default = None


@contextlib.contextmanager
def use_limiter(limiter: RateLimiter) -> Generator[RateLimiter]:
    """Temporarily install `limiter` as the process default.

    The driver's `run_claude` calls `get_default_limiter()`; tests that
    want a controlled tmp-dir-backed limiter use this contextmanager
    to swap one in without touching env vars or the on-disk default
    state.
    """
    global _default
    previous = _default
    _default = limiter
    try:
        yield limiter
    finally:
        _default = previous
