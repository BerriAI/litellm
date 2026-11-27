import asyncio

import pytest

import litellm
from litellm import DualCache, Router
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.utils import hash_token
from tests.unit.proxy.hooks.native_capacity_handler import NativeDynamicRateLimitHandler as DynamicRateLimitHandler
from tests.unit.proxy.hooks.native_capacity_handler import pre_call_at_saturation


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
async def test_dynamic_rate_limiter_v3_uses_atomic_check_and_increment(monkeypatch: pytest.MonkeyPatch):
    """
    Regression test: dynamic limiter's enforced descriptors flow through
    the atomic check-and-increment script, not the legacy read_only=True
    check followed by a separate read_only=False increment.

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
    redis = handler.saturation_redis()

    user = UserAPIKeyAuth(api_key=hash_token("dyn-atomic-key"))
    user.metadata = {"priority": "high"}

    await pre_call_at_saturation(handler, user, model, saturation=0.0)

    atomic_descriptors_observed: list[list[str]] = redis.atomic_descriptor_keys()
    assert atomic_descriptors_observed, (
        "Dynamic limiter must route enforced descriptors through "
        "the atomic check-and-increment script (no legacy read_only=True / "
        "separate-increment pattern)."
    )
    assert "model_saturation_check" in atomic_descriptors_observed[0], (
        f"Expected model_saturation_check in atomic descriptor set. Got: {atomic_descriptors_observed}"
    )
