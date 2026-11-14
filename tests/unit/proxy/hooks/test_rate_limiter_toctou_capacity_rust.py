import asyncio
from typing import List

import pytest

import litellm
from litellm import DualCache, Router
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.utils import hash_token
from tests.unit.proxy.hooks.native_capacity_handler import NativeDynamicRateLimitHandler as DynamicRateLimitHandler
from tests.unit.proxy.hooks.native_capacity_handler import pre_call_at_saturation


@pytest.fixture(autouse=True)
def native_dynamic_rate_limiter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_NATIVE_DYNAMIC_RATE_LIMITER_V3", "true")


def _make_phase1_barrier(num_concurrent: int, timeout: float = 0.1):
    """
    Sync primitive that, pre-fix, forces all N concurrent coroutines to finish
    their read-only Phase 1 check before any proceeds to Phase 3 increment —
    mimicking asyncio I/O scheduling under load on the vulnerable code.

    Wraps `should_rate_limit` so on `read_only=True` calls it waits until N
    callers arrive (TOCTOU window opened) OR `timeout` elapses (post-fix path:
    the limiter's serialization lock prevents N from ever reaching the
    barrier; the timeout lets the holder proceed so the lock can do its job).

    Pre-fix: barrier fills before timeout → all see same state → bypass observed.
    Post-fix: only lock-holder reaches barrier → times out → serial execution
    enforces limit.
    """
    arrived = 0
    all_arrived = asyncio.Event()

    def wrap(original):
        async def patched(*args, **kwargs):
            result = await original(*args, **kwargs)
            if kwargs.get("read_only"):
                nonlocal arrived
                arrived += 1
                if arrived >= num_concurrent:
                    all_arrived.set()
                try:
                    await asyncio.wait_for(all_arrived.wait(), timeout=timeout)
                except asyncio.TimeoutError:
                    pass
            return result

        return patched

    return wrap


@pytest.mark.asyncio
async def test_dynamic_rate_limiter_v3_concurrent_bypasses_model_capacity(monkeypatch):
    """
    DynamicRateLimitHandler PHASE 1 (read_only check) → PHASE 3 (increment)
    is non-atomic: dynamic_rate_limiter_v3.py:463-548.

    With TPM=100 model capacity and 5 concurrent priority="high" requests
    each consuming the full model_saturation_check counter, all observe the
    same Phase 1 state (counter=0), all pass, all proceed to Phase 3.

    Sequential atomic semantics would block requests once the model counter
    reaches its limit. TOCTOU lets all pass Phase 1 simultaneously.
    """
    NUM_CONCURRENT = 10
    MODEL_RPM = 2
    # Sequential bound: dynamic limiter rejects when `counter > current_limit`
    # (strict `>`), so a request whose Phase 1 sees counter=RPM still passes
    # (RPM is not strictly greater). Atomic execution therefore admits up to
    # RPM + 1 successes before the next sees counter > RPM.
    MAX_SEQUENTIAL_SUCCESSES = MODEL_RPM + 1

    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")
    monkeypatch.setattr(litellm, "priority_reservation", {"high": 0.9, "low": 0.1})

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

    model = "toctou-dyn-model"
    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "api_base": "test-base",
                    "rpm": MODEL_RPM,
                },
            }
        ]
    )
    handler.update_variables(llm_router=llm_router)

    barrier = _make_phase1_barrier(NUM_CONCURRENT)
    handler.v3_limiter.should_rate_limit = barrier(handler.v3_limiter.should_rate_limit)

    async def one_request(idx: int):
        user = UserAPIKeyAuth(api_key=hash_token(f"dyn-key-{idx}"))
        user.metadata = {"priority": "high"}
        try:
            await pre_call_at_saturation(handler, user, model, saturation=0.0)
            return "OK"
        except Exception as e:
            return e

    results = await asyncio.gather(
        *[one_request(i) for i in range(NUM_CONCURRENT)],
        return_exceptions=True,
    )
    successes = [r for r in results if r == "OK"]

    assert len(successes) <= MAX_SEQUENTIAL_SUCCESSES, (
        f"TOCTOU bypass in DynamicRateLimitHandler: {len(successes)}/{NUM_CONCURRENT} "
        f"concurrent requests passed Phase 1 + Phase 3 despite model RPM={MODEL_RPM}. "
        f"Atomic check-and-increment would block once counter > RPM "
        f"(at most {MAX_SEQUENTIAL_SUCCESSES} sequential successes)."
    )


@pytest.mark.asyncio
async def test_dynamic_rate_limiter_v3_uses_atomic_check_and_increment(monkeypatch):
    """
    Regression test: dynamic limiter's enforced descriptors flow through
    `atomic_check_and_increment_by_n`, not the legacy
    read_only=True check followed by a separate read_only=False increment.

    When priority is enforced (saturation >= threshold), priority_model is
    bundled into the atomic call alongside model_saturation_check. When not
    enforced, priority counter is incremented for tracking only.
    """
    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")
    monkeypatch.setattr(litellm, "priority_reservation", {"high": 0.9, "low": 0.1})

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

    model = "atomic-dyn-model"
    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "api_base": "test-base",
                    "tpm": 1000,
                },
            }
        ]
    )
    handler.update_variables(llm_router=llm_router)

    atomic_descriptors_observed: List[List[str]] = []
    original_atomic = handler.v3_limiter.atomic_check_and_increment_by_n

    async def logging_atomic(*args, **kwargs):
        ds = kwargs.get("descriptors") or (args[0] if args else [])
        atomic_descriptors_observed.append([d["key"] for d in ds])
        return await original_atomic(*args, **kwargs)

    handler.v3_limiter.atomic_check_and_increment_by_n = logging_atomic

    user = UserAPIKeyAuth(api_key=hash_token("dyn-atomic-key"))
    user.metadata = {"priority": "high"}

    await pre_call_at_saturation(handler, user, model, saturation=0.0)

    assert atomic_descriptors_observed, (
        "Dynamic limiter must route enforced descriptors through "
        "atomic_check_and_increment_by_n (no legacy read_only=True / "
        "separate-increment pattern)."
    )
    assert "model_saturation_check" in atomic_descriptors_observed[0], (
        f"Expected model_saturation_check in atomic descriptor set. Got: {atomic_descriptors_observed}"
    )


@pytest.mark.asyncio
async def test_dynamic_rate_limiter_v3_fails_closed_on_unknown_descriptor(monkeypatch):
    """
    Fail-closed guard: when atomic_check_and_increment_by_n returns
    overall_code=OVER_LIMIT but with a descriptor_key the dispatcher does
    not recognize, the dynamic limiter must raise 429 rather than silently
    fall through.

    Reproduces by patching atomic_check_and_increment_by_n to return an
    OVER_LIMIT response carrying an unknown descriptor_key.
    """
    from fastapi import HTTPException

    monkeypatch.setenv("LITELLM_LICENSE", "test-license-key")
    monkeypatch.setattr(litellm, "priority_reservation", {"high": 0.9, "low": 0.1})

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

    model = "fail-closed-model"
    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "test-key",
                    "api_base": "test-base",
                    "tpm": 1000,
                },
            }
        ]
    )
    handler.update_variables(llm_router=llm_router)

    async def fake_atomic(*args, **kwargs):
        return {
            "overall_code": "OVER_LIMIT",
            "statuses": [
                {
                    "code": "OVER_LIMIT",
                    "current_limit": 100,
                    "limit_remaining": 0,
                    "rate_limit_type": "tokens",
                    "descriptor_key": "future_unrecognized_descriptor",
                }
            ],
        }

    handler.v3_limiter.atomic_check_and_increment_by_n = fake_atomic

    user = UserAPIKeyAuth(api_key=hash_token("fail-closed-key"))
    user.metadata = {"priority": "high"}

    with pytest.raises(HTTPException) as exc:
        await pre_call_at_saturation(handler, user, model, saturation=0.0)
    assert exc.value.status_code == 429, f"Expected 429 fail-closed on unknown descriptor; got {exc.value.status_code}"
