"""Unit tests for the cross-process token-bucket rate limiter.

The tests cover three layers:

1. Provider inference from model alias — the matrix-column mapping the
   live tests rely on (`-bedrock-converse` vs `-bedrock-invoke` vs
   `-azure` vs `-vertex` vs bare = anthropic).

2. Config parsing — env-var precedence, fallback to default, malformed
   input handling, burst override semantics. These run against
   `os.environ`-shaped dicts so we don't have to monkeypatch globals.

3. Token-bucket behavior — enforcing rate, accumulating burst, never
   over-spending across a fake clock. Filesystem state is exercised
   with a real `tmp_path` because the persistence is the whole point;
   the only injected seam is `clock` (and `sleep`, so tests don't
   actually wait on wall time).

The cross-process flock semantics are exercised indirectly: every
test creates a fresh `RateLimiter` rooted at `tmp_path`, so the same
file lock that protects production is exercised here too. We don't
fork to test multi-process behavior in this file because pytest
fixtures + xdist already do that for the integration suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from claude_code.json_types import JSON_OBJECT_ADAPTER
from claude_code.rate_limiter import (
    ALL_PROVIDERS,
    BURST_ENV,
    DEFAULT_RATE,
    PROVIDER_ANTHROPIC,
    PROVIDER_AZURE,
    PROVIDER_BEDROCK_CONVERSE,
    PROVIDER_BEDROCK_INVOKE,
    PROVIDER_VERTEX_AI,
    ProviderConfig,
    RateLimiter,
    get_default_limiter,
    infer_provider,
    load_config,
    reset_default_limiter,
    use_limiter,
)


# ---------------------------------------------------------------------------
# Provider inference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model, expected",
    [
        ("claude-haiku-4-5", PROVIDER_ANTHROPIC),
        ("claude-sonnet-4-6", PROVIDER_ANTHROPIC),
        ("claude-opus-4-7", PROVIDER_ANTHROPIC),
        ("claude-haiku-4-5-azure", PROVIDER_AZURE),
        ("claude-sonnet-4-6-azure", PROVIDER_AZURE),
        ("claude-opus-4-7-vertex", PROVIDER_VERTEX_AI),
        ("claude-haiku-4-5-bedrock-converse", PROVIDER_BEDROCK_CONVERSE),
        ("claude-haiku-4-5-bedrock-invoke", PROVIDER_BEDROCK_INVOKE),
    ],
)
def test_infer_provider_maps_alias_suffix_to_column(model: str, expected: str) -> None:
    assert infer_provider(model) == expected


def test_infer_provider_bedrock_converse_beats_bedrock_invoke_lookup_order() -> None:
    """Both bedrock suffixes contain `bedrock`; the more-specific suffix wins."""
    assert infer_provider("claude-foo-bedrock-converse") == PROVIDER_BEDROCK_CONVERSE
    assert infer_provider("claude-foo-bedrock-invoke") == PROVIDER_BEDROCK_INVOKE


def test_infer_provider_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        infer_provider("")


def test_infer_provider_is_case_insensitive() -> None:
    """Aliases in the proxy config sometimes drift between cases; we
    should still route them to the right column."""
    assert infer_provider("CLAUDE-OPUS-4-7-AZURE") == PROVIDER_AZURE


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def test_load_config_uses_default_rate_when_env_absent() -> None:
    cfg = load_config(env={})
    for provider in ALL_PROVIDERS:
        assert cfg[provider].rate_per_sec == DEFAULT_RATE
        assert cfg[provider].burst == DEFAULT_RATE


def test_load_config_reads_per_provider_rate() -> None:
    cfg = load_config(
        env={
            "LITELLM_COMPAT_RATE_ANTHROPIC": "10",
            "LITELLM_COMPAT_RATE_AZURE": "0.5",
        }
    )
    assert cfg[PROVIDER_ANTHROPIC].rate_per_sec == 10.0
    assert cfg[PROVIDER_AZURE].rate_per_sec == 0.5
    assert cfg[PROVIDER_VERTEX_AI].rate_per_sec == DEFAULT_RATE


def test_load_config_zero_rate_disables_provider() -> None:
    cfg = load_config(env={"LITELLM_COMPAT_RATE_BEDROCK_INVOKE": "0"})
    assert cfg[PROVIDER_BEDROCK_INVOKE].enabled is False


def test_load_config_burst_override_applies_to_every_provider() -> None:
    cfg = load_config(
        env={
            "LITELLM_COMPAT_RATE_ANTHROPIC": "5",
            BURST_ENV: "20",
        }
    )
    for provider in ALL_PROVIDERS:
        assert cfg[provider].burst == 20.0


def test_load_config_falls_back_on_malformed_value() -> None:
    cfg = load_config(env={"LITELLM_COMPAT_RATE_ANTHROPIC": "not-a-number"})
    assert cfg[PROVIDER_ANTHROPIC].rate_per_sec == DEFAULT_RATE


def test_load_config_burst_floors_at_one_when_rate_is_low() -> None:
    """A 0.5/s rate with no burst override must still allow at least
    one immediate request — otherwise the very first call would block."""
    cfg = load_config(env={"LITELLM_COMPAT_RATE_ANTHROPIC": "0.5"})
    assert cfg[PROVIDER_ANTHROPIC].burst == 1.0


# ---------------------------------------------------------------------------
# Token bucket
# ---------------------------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.now: float = 1_000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def fake_clock() -> _Clock:
    """A controllable monotonic clock + sleep for the limiter under test.

    Tests advance `clock.now` to simulate elapsed wall time. `sleep`
    adds the requested duration to `clock.now` instead of actually
    sleeping, so a "wait 200ms" code path runs in microseconds and
    is deterministic.
    """
    return _Clock()


def _make_limiter(
    tmp_path: Path, fake_clock: _Clock, *, rate: float = 10.0, burst: float | None = None
) -> RateLimiter:
    cfg = {
        p: ProviderConfig(rate_per_sec=rate, burst=burst if burst is not None else rate)
        for p in ALL_PROVIDERS
    }
    return RateLimiter(
        config=cfg,
        state_dir=tmp_path,
        clock=fake_clock,
        sleep=fake_clock.sleep,
    )


def test_acquire_first_call_does_not_wait(tmp_path: Path, fake_clock: _Clock) -> None:
    """A freshly-initialized bucket starts full; the first acquire is free."""
    limiter = _make_limiter(tmp_path, fake_clock, rate=10.0, burst=10.0)
    waited = limiter.acquire(PROVIDER_ANTHROPIC)
    assert waited == 0.0
    assert fake_clock.sleeps == []


def test_acquire_disabled_provider_returns_immediately(tmp_path: Path, fake_clock: _Clock) -> None:
    """rate=0 ⇒ no throttling, even if every other provider is throttled."""
    cfg = {p: ProviderConfig(rate_per_sec=0.0, burst=0.0) for p in ALL_PROVIDERS}
    limiter = RateLimiter(
        config=cfg, state_dir=tmp_path, clock=fake_clock, sleep=fake_clock.sleep
    )
    for _ in range(100):
        assert limiter.acquire(PROVIDER_ANTHROPIC) == 0.0
    assert fake_clock.sleeps == []


def test_acquire_burns_through_burst_then_throttles(tmp_path: Path, fake_clock: _Clock) -> None:
    """`burst` immediate requests succeed; the next one waits 1/rate seconds."""
    limiter = _make_limiter(tmp_path, fake_clock, rate=2.0, burst=3.0)

    for _ in range(3):
        assert limiter.acquire(PROVIDER_ANTHROPIC) == 0.0

    # Bucket is empty; next call must sleep ~0.5s to earn one token at 2/s.
    waited = limiter.acquire(PROVIDER_ANTHROPIC)
    assert waited == pytest.approx(0.5, abs=0.01)  # pyright: ignore[reportUnknownMemberType]  # untyped upstream


def test_acquire_refills_with_elapsed_time(tmp_path: Path, fake_clock: _Clock) -> None:
    """Advancing the clock between calls credits tokens at the configured rate."""
    limiter = _make_limiter(tmp_path, fake_clock, rate=4.0, burst=1.0)

    assert limiter.acquire(PROVIDER_ANTHROPIC) == 0.0  # consumes the 1-token burst
    fake_clock.now += 0.25  # 0.25s × 4/s = 1 token earned
    assert limiter.acquire(PROVIDER_ANTHROPIC) == 0.0


def test_acquire_caps_refill_at_burst(tmp_path: Path, fake_clock: _Clock) -> None:
    """A long quiet period must not let the bucket grow past `burst`."""
    limiter = _make_limiter(tmp_path, fake_clock, rate=10.0, burst=2.0)

    fake_clock.now += 1_000  # would earn 10_000 tokens uncapped
    # Only `burst` (=2) immediate calls should succeed before throttling.
    assert limiter.acquire(PROVIDER_ANTHROPIC) == 0.0
    assert limiter.acquire(PROVIDER_ANTHROPIC) == 0.0
    waited = limiter.acquire(PROVIDER_ANTHROPIC)
    assert waited > 0


def test_acquire_independent_buckets_per_provider(tmp_path: Path, fake_clock: _Clock) -> None:
    """Anthropic exhaustion must not throttle Azure (each column has its own bucket)."""
    limiter = _make_limiter(tmp_path, fake_clock, rate=2.0, burst=1.0)

    assert limiter.acquire(PROVIDER_ANTHROPIC) == 0.0
    # Anthropic bucket is now empty; Azure is untouched.
    assert limiter.acquire(PROVIDER_AZURE) == 0.0


def test_acquire_persists_state_across_limiter_instances(tmp_path: Path) -> None:
    """A fresh RateLimiter must read the on-disk state, not start fresh.

    This is the property that makes the limiter cross-process: an
    xdist worker created mid-run sees the credit consumed by other
    workers, instead of getting its own private bucket.
    """
    cfg = {p: ProviderConfig(rate_per_sec=10.0, burst=2.0) for p in ALL_PROVIDERS}
    shared_clock = _Clock()

    first = RateLimiter(
        config=cfg, state_dir=tmp_path, clock=shared_clock, sleep=shared_clock.sleep
    )
    first.acquire(PROVIDER_ANTHROPIC)
    first.acquire(PROVIDER_ANTHROPIC)
    # bucket is now empty

    second = RateLimiter(
        config=cfg, state_dir=tmp_path, clock=shared_clock, sleep=shared_clock.sleep
    )
    waited = second.acquire(PROVIDER_ANTHROPIC)
    assert waited > 0  # had to wait, didn't see a fresh full bucket


def test_acquire_recovers_from_corrupt_state_file(tmp_path: Path, fake_clock: _Clock) -> None:
    """A truncated/garbage state file must not crash the test session."""
    state_file = tmp_path / f"{PROVIDER_ANTHROPIC}.json"
    state_file.write_text("not-json {{")

    limiter = _make_limiter(tmp_path, fake_clock, rate=5.0, burst=5.0)
    assert limiter.acquire(PROVIDER_ANTHROPIC) == 0.0


def test_acquire_handles_clock_going_backward(tmp_path: Path, fake_clock: _Clock) -> None:
    """Across a host suspend/resume the monotonic clock can briefly
    go backward; we must not interpret that as removing tokens."""
    limiter = _make_limiter(tmp_path, fake_clock, rate=1.0, burst=2.0)
    limiter.acquire(PROVIDER_ANTHROPIC)
    fake_clock.now -= 10  # clock moved backward
    # Bucket should still have ~1 token left from the burst, not -9.
    assert limiter.acquire(PROVIDER_ANTHROPIC) == 0.0


# ---------------------------------------------------------------------------
# Process-default singleton
# ---------------------------------------------------------------------------


def test_use_limiter_swaps_default_for_block(tmp_path: Path) -> None:
    sentinel_cfg = {
        p: ProviderConfig(rate_per_sec=0.0, burst=0.0) for p in ALL_PROVIDERS
    }
    sentinel = RateLimiter(config=sentinel_cfg, state_dir=tmp_path)
    reset_default_limiter()
    try:
        with use_limiter(sentinel):
            assert get_default_limiter() is sentinel
        # After the context exits, the default goes back to whatever it
        # was — in this test that's "rebuilt on next access" because we
        # called reset_default_limiter() above.
        assert get_default_limiter() is not sentinel
    finally:
        reset_default_limiter()


# ---------------------------------------------------------------------------
# Persistence shape
# ---------------------------------------------------------------------------


def test_state_file_is_json_after_acquire(tmp_path: Path, fake_clock: _Clock) -> None:
    limiter = _make_limiter(tmp_path, fake_clock, rate=5.0, burst=5.0)
    limiter.acquire(PROVIDER_ANTHROPIC)
    state_file = tmp_path / f"{PROVIDER_ANTHROPIC}.json"
    payload = JSON_OBJECT_ADAPTER.validate_json(state_file.read_text())
    assert "tokens" in payload
    assert "last_refill" in payload
    assert payload["tokens"] == pytest.approx(4.0)  # pyright: ignore[reportUnknownMemberType]  # untyped upstream
