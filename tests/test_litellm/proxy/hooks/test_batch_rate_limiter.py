"""
Tests for `tpd_limit` (tokens per day) enforcement on batch submissions.

A batch's rows are scheduled by the provider, so a caller cannot keep a large
batch under a per-minute RPM/TPM budget. Scopes that configure `tpd_limit`
are charged against a 24h token window instead of their minute counters.
"""

import time
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Final

import pytest
from fastapi import HTTPException

from litellm import DualCache
from litellm.constants import BATCH_TPD_WINDOW_SECONDS
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.batch_rate_limiter import BatchFileUsage
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _PROXY_MaxParallelRequestsHandler_v3,
)
from litellm.proxy.utils import InternalUsageCache, hash_token


class _Clock:
    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now


def _make_limiters(clock: _Clock | None = None):
    internal_usage_cache = InternalUsageCache(dual_cache=DualCache())
    rate_limiter = _PROXY_MaxParallelRequestsHandler_v3(internal_usage_cache=internal_usage_cache, time_provider=clock)
    batch_limiter = rate_limiter._get_batch_rate_limiter()
    assert batch_limiter is not None
    return internal_usage_cache, rate_limiter, batch_limiter


async def _counter(internal_usage_cache, rate_limiter, descriptor_key, value, rate_limit_type):
    cache_key = rate_limiter.create_rate_limit_keys(descriptor_key, value, rate_limit_type)
    raw = await internal_usage_cache.async_get_cache(key=cache_key, litellm_parent_otel_span=None, local_only=True)
    return int(raw or 0)


@pytest.mark.asyncio
async def test_batch_over_rpm_and_tpm_but_under_tpd_is_accepted():
    internal_usage_cache, rate_limiter, batch_limiter = _make_limiters()
    api_key = hash_token("tpd-key")
    user_api_key_dict = UserAPIKeyAuth(api_key=api_key, rpm_limit=1, tpm_limit=10, tpd_limit=1000)

    await batch_limiter._check_and_increment_batch_counters(
        user_api_key_dict=user_api_key_dict,
        data={},
        batch_usage=BatchFileUsage(total_tokens=500, request_count=50),
    )

    assert await _counter(internal_usage_cache, rate_limiter, "api_key_tpd", api_key, "tokens") == 500
    assert await _counter(internal_usage_cache, rate_limiter, "api_key", api_key, "requests") == 0
    assert await _counter(internal_usage_cache, rate_limiter, "api_key", api_key, "tokens") == 0


@pytest.mark.asyncio
async def test_cumulative_batch_tokens_over_tpd_returns_429_with_remaining_daily_window():
    window_start = datetime(2026, 9, 13, 8, 0, 0)
    clock = _Clock(window_start)
    _internal_usage_cache, _rate_limiter, batch_limiter = _make_limiters(clock)
    user_api_key_dict = UserAPIKeyAuth(api_key=hash_token("tpd-key-2"), rpm_limit=1, tpd_limit=1000)

    await batch_limiter._check_and_increment_batch_counters(
        user_api_key_dict=user_api_key_dict,
        data={},
        batch_usage=BatchFileUsage(total_tokens=600, request_count=6),
    )
    clock.now = datetime(2026, 9, 13, 11, 0, 0)
    with pytest.raises(HTTPException) as exc:
        await batch_limiter._check_and_increment_batch_counters(
            user_api_key_dict=user_api_key_dict,
            data={},
            batch_usage=BatchFileUsage(total_tokens=600, request_count=6),
        )

    assert exc.value.status_code == 429
    assert "api_key_tpd" in str(exc.value.detail)
    assert "600 tokens but only 400 tokens remaining out of 1000 TPD limit" in str(exc.value.detail)
    assert exc.value.headers["retry-after"] == str(BATCH_TPD_WINDOW_SECONDS - 3 * 3600)
    assert exc.value.headers["reset_at"] == "2026-09-14 08:00:00 UTC"


@pytest.mark.asyncio
async def test_failed_batch_submission_refunds_tpd_tokens():
    internal_usage_cache, rate_limiter, batch_limiter = _make_limiters()
    api_key = hash_token("tpd-refund-key")
    user_api_key_dict = UserAPIKeyAuth(api_key=api_key, rpm_limit=1, tpd_limit=1000)

    await batch_limiter._check_and_increment_batch_counters(
        user_api_key_dict=user_api_key_dict,
        data={},
        batch_usage=BatchFileUsage(total_tokens=600, request_count=6),
    )
    await rate_limiter.async_post_call_failure_hook(
        request_data={},
        original_exception=RuntimeError("provider rejected the file"),
        user_api_key_dict=user_api_key_dict,
    )

    assert await _counter(internal_usage_cache, rate_limiter, "api_key_tpd", api_key, "tokens") == 0
    await batch_limiter._check_and_increment_batch_counters(
        user_api_key_dict=user_api_key_dict,
        data={},
        batch_usage=BatchFileUsage(total_tokens=1000, request_count=10),
    )
    assert await _counter(internal_usage_cache, rate_limiter, "api_key_tpd", api_key, "tokens") == 1000


@pytest.mark.asyncio
async def test_tpd_refund_applies_once_and_only_to_daily_counters():
    internal_usage_cache, rate_limiter, batch_limiter = _make_limiters()
    team_key = UserAPIKeyAuth(
        api_key=hash_token("tpd-refund-team-key"),
        rpm_limit=100,
        tpm_limit=10_000,
        team_id="team-r",
        team_tpd_limit=5000,
    )

    await batch_limiter._check_and_increment_batch_counters(
        user_api_key_dict=team_key,
        data={},
        batch_usage=BatchFileUsage(total_tokens=800, request_count=8),
    )
    await rate_limiter.async_post_call_failure_hook(
        request_data={}, original_exception=RuntimeError("boom"), user_api_key_dict=team_key
    )
    await rate_limiter.async_post_call_failure_hook(
        request_data={}, original_exception=RuntimeError("boom"), user_api_key_dict=team_key
    )

    assert await _counter(internal_usage_cache, rate_limiter, "team_tpd", "team-r", "tokens") == 0
    assert await _counter(internal_usage_cache, rate_limiter, "api_key", team_key.api_key, "tokens") == 800
    assert await _counter(internal_usage_cache, rate_limiter, "api_key", team_key.api_key, "requests") == 8


@pytest.mark.asyncio
async def test_rejected_batch_leaves_nothing_to_refund():
    internal_usage_cache, rate_limiter, batch_limiter = _make_limiters()
    api_key = hash_token("tpd-rejected-key")
    user_api_key_dict = UserAPIKeyAuth(api_key=api_key, tpd_limit=100)

    await batch_limiter._check_and_increment_batch_counters(
        user_api_key_dict=user_api_key_dict, data={}, batch_usage=BatchFileUsage(total_tokens=90, request_count=9)
    )
    with pytest.raises(HTTPException):
        await batch_limiter._check_and_increment_batch_counters(
            user_api_key_dict=user_api_key_dict, data={}, batch_usage=BatchFileUsage(total_tokens=20, request_count=2)
        )
    await rate_limiter.async_post_call_failure_hook(
        request_data={}, original_exception=RuntimeError("429 bubbled up"), user_api_key_dict=user_api_key_dict
    )

    assert await _counter(internal_usage_cache, rate_limiter, "api_key_tpd", api_key, "tokens") == 90


@pytest.mark.asyncio
async def test_batch_without_tpd_still_enforces_minute_rpm():
    _internal_usage_cache, _rate_limiter, batch_limiter = _make_limiters()
    user_api_key_dict = UserAPIKeyAuth(api_key=hash_token("rpm-only-key"), rpm_limit=1, tpm_limit=1000)

    with pytest.raises(HTTPException) as exc:
        await batch_limiter._check_and_increment_batch_counters(
            user_api_key_dict=user_api_key_dict,
            data={},
            batch_usage=BatchFileUsage(total_tokens=50, request_count=5),
        )

    assert exc.value.status_code == 429
    assert "RPM limit" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_team_tpd_replaces_team_minute_limits_but_key_minute_limits_still_apply():
    internal_usage_cache, rate_limiter, batch_limiter = _make_limiters()
    team_key = UserAPIKeyAuth(
        api_key=hash_token("team-key"),
        team_id="team-1",
        team_rpm_limit=1,
        team_tpm_limit=10,
        team_tpd_limit=5000,
    )

    await batch_limiter._check_and_increment_batch_counters(
        user_api_key_dict=team_key,
        data={},
        batch_usage=BatchFileUsage(total_tokens=800, request_count=8),
    )
    assert await _counter(internal_usage_cache, rate_limiter, "team_tpd", "team-1", "tokens") == 800
    assert await _counter(internal_usage_cache, rate_limiter, "team", "team-1", "requests") == 0

    key_rpm_in_team_with_tpd = UserAPIKeyAuth(
        api_key=hash_token("team-key-2"),
        rpm_limit=1,
        team_id="team-1",
        team_tpd_limit=5000,
    )
    with pytest.raises(HTTPException) as exc:
        await batch_limiter._check_and_increment_batch_counters(
            user_api_key_dict=key_rpm_in_team_with_tpd,
            data={},
            batch_usage=BatchFileUsage(total_tokens=10, request_count=2),
        )
    assert exc.value.status_code == 429
    assert "api_key:" in str(exc.value.detail)
    assert "RPM limit" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_end_user_tpd_is_enforced_per_end_user():
    _internal_usage_cache, _rate_limiter, batch_limiter = _make_limiters()
    first_customer = UserAPIKeyAuth(
        api_key=hash_token("shared-key"), end_user_id="customer-a", end_user_rpm_limit=1, end_user_tpd_limit=100
    )
    second_customer = UserAPIKeyAuth(
        api_key=hash_token("shared-key"), end_user_id="customer-b", end_user_rpm_limit=1, end_user_tpd_limit=100
    )

    await batch_limiter._check_and_increment_batch_counters(
        user_api_key_dict=first_customer, data={}, batch_usage=BatchFileUsage(total_tokens=90, request_count=9)
    )
    await batch_limiter._check_and_increment_batch_counters(
        user_api_key_dict=second_customer, data={}, batch_usage=BatchFileUsage(total_tokens=90, request_count=9)
    )
    with pytest.raises(HTTPException) as exc:
        await batch_limiter._check_and_increment_batch_counters(
            user_api_key_dict=first_customer, data={}, batch_usage=BatchFileUsage(total_tokens=20, request_count=2)
        )
    assert exc.value.status_code == 429
    assert "end_user_tpd: customer-a" in str(exc.value.detail)


def test_tpd_only_key_is_not_skipped_as_having_no_limits():
    _internal_usage_cache, _rate_limiter, batch_limiter = _make_limiters()
    descriptors = batch_limiter._create_batch_rate_limit_descriptors(
        user_api_key_dict=UserAPIKeyAuth(api_key=hash_token("tpd-only"), tpd_limit=100),
        data={},
    )
    assert batch_limiter._has_applicable_batch_rate_limits(descriptors) is True


def test_online_descriptors_ignore_tpd_limit():
    _internal_usage_cache, rate_limiter, _batch_limiter = _make_limiters()
    api_key = hash_token("online-key")
    descriptors = rate_limiter._create_rate_limit_descriptors(
        user_api_key_dict=UserAPIKeyAuth(api_key=api_key, rpm_limit=5, tpd_limit=100, team_id="t", team_tpd_limit=9),
        data={"model": "gpt-4o"},
        rpm_limit_type=None,
        tpm_limit_type=None,
        model_has_failures=False,
    )
    assert [(d["key"], d["rate_limit"]["window_size"]) for d in descriptors] == [("api_key", rate_limiter.window_size)]


@pytest.fixture(params=["Europe/Paris", "Asia/Kolkata", "America/Los_Angeles"])
def process_timezone(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    monkeypatch.setenv("TZ", request.param)
    time.tzset()
    yield request.param
    monkeypatch.undo()
    time.tzset()


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="switching the process timezone needs time.tzset()")
@pytest.mark.asyncio
async def test_batch_rate_limit_error_reports_reset_time_in_utc_on_a_non_utc_proxy(process_timezone: str) -> None:
    window_start: Final = datetime(2026, 9, 13, 8, 0, 0, tzinfo=timezone.utc)
    clock: Final = _Clock(window_start)
    _internal_usage_cache, _rate_limiter, batch_limiter = _make_limiters(clock)
    user_api_key_dict: Final = UserAPIKeyAuth(api_key=hash_token("tpd-key-utc"), rpm_limit=1, tpd_limit=1000)

    await batch_limiter._check_and_increment_batch_counters(
        user_api_key_dict=user_api_key_dict,
        data={},
        batch_usage=BatchFileUsage(total_tokens=600, request_count=6),
    )
    clock.now = datetime(2026, 9, 13, 11, 0, 0, tzinfo=timezone.utc)
    with pytest.raises(HTTPException) as exc:
        await batch_limiter._check_and_increment_batch_counters(
            user_api_key_dict=user_api_key_dict,
            data={},
            batch_usage=BatchFileUsage(total_tokens=600, request_count=6),
        )

    assert exc.value.status_code == 429
    assert exc.value.headers["retry-after"] == str(BATCH_TPD_WINDOW_SECONDS - 3 * 3600)
    assert exc.value.headers["reset_at"] == "2026-09-14 08:00:00 UTC"
    assert str(exc.value.detail).endswith("Limit resets at: 2026-09-14 08:00:00 UTC")
