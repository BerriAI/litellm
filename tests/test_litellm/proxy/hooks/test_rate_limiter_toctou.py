"""
Tests validating TOCTOU race condition in batch + dynamic rate limiters.

Issue: rate-limit check (read_only=True) and counter increment happen as two
separate awaits. Concurrent requests all observe the same pre-increment state,
all pass validation, then all increment — bypassing the limit.

Vulnerable code paths:
- litellm/proxy/hooks/batch_rate_limiter.py:181-248
  (_check_and_increment_batch_counters: should_rate_limit(read_only=True)
  → validate → async_increment_tokens_with_ttl_preservation)
- litellm/proxy/hooks/dynamic_rate_limiter_v3.py:463-548
  (_check_rate_limits PHASE 1 read_only check → PHASE 3 increment)

These tests EXPECTED to fail against current (vulnerable) code and pass once
check-and-increment becomes atomic.
"""

import asyncio
import os
import sys
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm import DualCache, Router
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.batch_rate_limiter import BatchFileUsage
from litellm.proxy.hooks.dynamic_rate_limiter_v3 import (
    _PROXY_DynamicRateLimitHandlerV3 as DynamicRateLimitHandler,
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _PROXY_MaxParallelRequestsHandler_v3,
)
from litellm.proxy.utils import InternalUsageCache, hash_token


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
async def test_batch_limiter_concurrent_bypasses_tpm_via_toctou():
    """
    5 concurrent batch submissions of 40 tokens each against TPM=100 limit.

    Sequential semantics: only 2 batches fit (2 * 40 = 80 ≤ 100, 3rd at 120 fails).
    With TOCTOU: all 5 succeed → 200 tokens consumed, 100% over limit.

    Demonstrates batch_rate_limiter.py:183-248 multi-phase flaw.
    """
    NUM_CONCURRENT = 5
    BATCH_TOKENS = 40
    TPM_LIMIT = 100

    dual_cache = DualCache()
    internal_usage_cache = InternalUsageCache(dual_cache=dual_cache)
    rate_limiter = _PROXY_MaxParallelRequestsHandler_v3(
        internal_usage_cache=internal_usage_cache
    )
    batch_limiter = rate_limiter._get_batch_rate_limiter()
    assert batch_limiter is not None

    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("toctou-batch-key"),
        tpm_limit=TPM_LIMIT,
        rpm_limit=1000,
    )
    batch_usage = BatchFileUsage(
        total_tokens=BATCH_TOKENS, request_count=1
    )

    barrier = _make_phase1_barrier(NUM_CONCURRENT)
    rate_limiter.should_rate_limit = barrier(rate_limiter.should_rate_limit)

    results = await asyncio.gather(
        *[
            batch_limiter._check_and_increment_batch_counters(
                user_api_key_dict=user_api_key_dict,
                data={},
                batch_usage=batch_usage,
            )
            for _ in range(NUM_CONCURRENT)
        ],
        return_exceptions=True,
    )

    successes = [r for r in results if not isinstance(r, Exception)]
    rejections = [r for r in results if isinstance(r, Exception)]
    total_consumed = len(successes) * BATCH_TOKENS
    max_allowed_successes = TPM_LIMIT // BATCH_TOKENS  # 2

    assert len(successes) <= max_allowed_successes, (
        f"TOCTOU bypass: {len(successes)}/{NUM_CONCURRENT} concurrent batches "
        f"passed despite TPM={TPM_LIMIT}. Consumed {total_consumed} tokens "
        f"({total_consumed - TPM_LIMIT} over limit). "
        f"Atomic check-and-increment would allow ≤{max_allowed_successes}. "
        f"Rejections: {len(rejections)}"
    )


@pytest.mark.asyncio
async def test_batch_limiter_check_and_increment_is_two_separate_calls():
    """
    Structural test: _check_and_increment_batch_counters issues a read_only=True
    check followed by a separate increment call — non-atomic by construction.

    Records call ordering on parallel_request_limiter to prove Phase 1 (check)
    and Phase 3 (increment) are not wrapped in a single Redis transaction /
    Lua script. After the fix, this pattern should be replaced with one
    atomic_check_and_increment call.
    """
    dual_cache = DualCache()
    internal_usage_cache = InternalUsageCache(dual_cache=dual_cache)
    rate_limiter = _PROXY_MaxParallelRequestsHandler_v3(
        internal_usage_cache=internal_usage_cache
    )
    batch_limiter = rate_limiter._get_batch_rate_limiter()
    assert batch_limiter is not None

    call_log: List[Dict[str, Any]] = []
    original_should = rate_limiter.should_rate_limit
    original_inc = rate_limiter.async_increment_tokens_with_ttl_preservation

    async def logging_should(*args, **kwargs):
        call_log.append(
            {"method": "should_rate_limit", "read_only": kwargs.get("read_only")}
        )
        return await original_should(*args, **kwargs)

    async def logging_inc(*args, **kwargs):
        call_log.append({"method": "async_increment_tokens_with_ttl_preservation"})
        return await original_inc(*args, **kwargs)

    rate_limiter.should_rate_limit = logging_should
    rate_limiter.async_increment_tokens_with_ttl_preservation = logging_inc

    user_api_key_dict = UserAPIKeyAuth(
        api_key=hash_token("structural-test-key"),
        tpm_limit=10000,
        rpm_limit=1000,
    )

    await batch_limiter._check_and_increment_batch_counters(
        user_api_key_dict=user_api_key_dict,
        data={},
        batch_usage=BatchFileUsage(total_tokens=50, request_count=1),
    )

    method_sequence = [c["method"] for c in call_log]
    assert "should_rate_limit" in method_sequence, "Expected Phase 1 check"
    phase1 = [
        c
        for c in call_log
        if c["method"] == "should_rate_limit" and c.get("read_only") is True
    ]
    phase3 = [
        c
        for c in call_log
        if c["method"] == "async_increment_tokens_with_ttl_preservation"
    ]
    assert len(phase1) >= 1 and len(phase3) >= 1, (
        f"Expected non-atomic Phase1+Phase3 pattern. call_log={call_log}"
    )
    phase1_idx = method_sequence.index("should_rate_limit")
    phase3_idx = method_sequence.index(
        "async_increment_tokens_with_ttl_preservation"
    )
    assert phase1_idx < phase3_idx, (
        "TOCTOU evidence: read-only check precedes increment as separate awaits — "
        "no atomic Lua script wraps both. Sequence: "
        f"{method_sequence}"
    )


@pytest.mark.asyncio
async def test_dynamic_rate_limiter_v3_concurrent_bypasses_model_capacity():
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

    os.environ["LITELLM_LICENSE"] = "test-license-key"
    litellm.priority_reservation = {"high": 0.9, "low": 0.1}

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

    from litellm.types.router import ModelGroupInfo

    model_group_info = ModelGroupInfo(
        model_group=model,
        providers=["openai"],
        rpm=MODEL_RPM,
        tpm=None,
    )

    async def one_request(idx: int):
        user = UserAPIKeyAuth(api_key=hash_token(f"dyn-key-{idx}"))
        user.metadata = {"priority": "high"}
        try:
            await handler._check_rate_limits(
                model=model,
                model_group_info=model_group_info,
                user_api_key_dict=user,
                priority="high",
                saturation=0.0,
                data={},
            )
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
async def test_dynamic_rate_limiter_v3_phase1_phase3_are_separate_awaits():
    """
    Structural proof of TOCTOU: dynamic_rate_limiter_v3._check_rate_limits
    issues a read_only=True call (Phase 1, line 464-468) followed by separate
    read_only=False calls (Phase 3, lines 526-530 + 534-538).

    Records each invocation of v3_limiter.should_rate_limit and asserts the
    Phase1→Phase3 sequence. After fix, both phases must collapse into a single
    atomic operation.
    """
    os.environ["LITELLM_LICENSE"] = "test-license-key"
    litellm.priority_reservation = {"high": 0.9, "low": 0.1}

    dual_cache = DualCache()
    handler = DynamicRateLimitHandler(internal_usage_cache=dual_cache)

    model = "structural-dyn-model"
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

    read_only_flags: List[Optional[bool]] = []
    original = handler.v3_limiter.should_rate_limit

    async def logging_should(*args, **kwargs):
        read_only_flags.append(kwargs.get("read_only"))
        return await original(*args, **kwargs)

    handler.v3_limiter.should_rate_limit = logging_should

    from litellm.types.router import ModelGroupInfo

    user = UserAPIKeyAuth(api_key=hash_token("dyn-structural-key"))
    user.metadata = {"priority": "high"}

    await handler._check_rate_limits(
        model=model,
        model_group_info=ModelGroupInfo(
            model_group=model,
            providers=["openai"],
            rpm=None,
            tpm=1000,
        ),
        user_api_key_dict=user,
        priority="high",
        saturation=0.0,
        data={},
    )

    assert True in read_only_flags or any(
        f is True for f in read_only_flags
    ), f"Expected read_only=True (Phase 1) call. Got: {read_only_flags}"
    assert any(f is False for f in read_only_flags), (
        f"Expected read_only=False (Phase 3) call. Got: {read_only_flags}"
    )

    first_read_only = next(
        (i for i, f in enumerate(read_only_flags) if f is True), None
    )
    first_write = next(
        (i for i, f in enumerate(read_only_flags) if f is False), None
    )
    assert first_read_only is not None and first_write is not None
    assert first_read_only < first_write, (
        f"TOCTOU evidence: Phase 1 (read_only) precedes Phase 3 (increment) "
        f"as separate non-atomic calls. read_only sequence: {read_only_flags}"
    )
