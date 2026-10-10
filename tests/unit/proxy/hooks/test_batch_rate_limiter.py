"""
Tests for `tpd_limit` (tokens per day) enforcement on batch submissions.

A batch's rows are scheduled by the provider, so a caller cannot keep a large
batch under a per-minute RPM/TPM budget. Scopes that configure `tpd_limit`
are charged against a 24h token window instead of their minute counters.
"""

import json
import time
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone
from typing import Final, Literal

import httpx
import pytest
import respx
from fastapi import HTTPException
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm import DualCache
from litellm.constants import BATCH_TPD_WINDOW_SECONDS
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.batch_rate_limiter import BatchFileUsage
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    PROXY_MaxParallelRequestsHandler_v3,
)
from litellm.proxy.utils import InternalUsageCache, hash_token
from litellm.types.llms.openai import ChatCompletionUserMessage, LiteLLMBatchCreateRequest


class _Clock:
    def __init__(self, start: datetime):
        self.now = start

    def __call__(self) -> datetime:
        return self.now


def _make_limiters(clock: _Clock | None = None):
    internal_usage_cache = InternalUsageCache(dual_cache=DualCache())
    rate_limiter = PROXY_MaxParallelRequestsHandler_v3(internal_usage_cache=internal_usage_cache, time_provider=clock)
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
    descriptors = rate_limiter.create_rate_limit_descriptors(
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


_BATCH_MODEL: Final = "gpt-3.5-turbo"
_MANAGED_FILE_ID: Final = (
    "bGl0ZWxsbV9wcm94eTphcHBsaWNhdGlvbi9vY3RldC1zdHJlYW07dW5pZmllZF9pZCxyZWdyZXNzaW9uLXRlc3QtZmlsZQ=="
)


class _BatchBody(TypedDict):
    model: ReadOnly[str]
    messages: ReadOnly[Sequence[ChatCompletionUserMessage]]


class _BatchLine(TypedDict):
    custom_id: ReadOnly[str]
    method: ReadOnly[Literal["POST"]]
    url: ReadOnly[Literal["/v1/chat/completions"]]
    body: ReadOnly[_BatchBody]


@pytest.fixture
def openai_files(monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter) -> respx.MockRouter:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    return respx_mock


def _batch_rows(messages: Sequence[str]) -> tuple[_BatchLine, ...]:
    return tuple(
        _BatchLine(
            custom_id=f"request-{i}",
            method="POST",
            url="/v1/chat/completions",
            body=_BatchBody(model=_BATCH_MODEL, messages=(ChatCompletionUserMessage(role="user", content=message),)),
        )
        for i, message in enumerate(messages, start=1)
    )


def _serve_file(router: respx.MockRouter, file_id: str, rows: Sequence[_BatchLine]) -> respx.Route:
    jsonl: Final = "\n".join(json.dumps(row) for row in rows)
    return router.get(f"https://api.openai.com/v1/files/{file_id}/content").mock(
        return_value=httpx.Response(200, content=jsonl.encode())
    )


def _token_counter_total(rows: Sequence[_BatchLine]) -> int:
    return sum(litellm.token_counter(model=row["body"]["model"], messages=row["body"]["messages"]) for row in rows)


def _create_batch_data(input_file_id: str) -> LiteLLMBatchCreateRequest:
    return LiteLLMBatchCreateRequest(model=_BATCH_MODEL, input_file_id=input_file_id)


@pytest.mark.asyncio
async def test_count_input_file_usage_matches_token_counter(openai_files: respx.MockRouter):
    _, _, batch_limiter = _make_limiters()
    rows: Final = _batch_rows(("Hello", "Hi there", "Hey"))
    content_route: Final = _serve_file(openai_files, "file-abc123", rows)

    usage: Final = await batch_limiter.count_input_file_usage(file_id="file-abc123", custom_llm_provider="openai")

    assert content_route.call_count == 1
    assert usage.request_count == 3
    assert usage.total_tokens == _token_counter_total(rows)


@pytest.mark.asyncio
async def test_batch_rate_limit_single_file_under_and_over_tpm(openai_files: respx.MockRouter):
    small_rows: Final = _batch_rows(("Hello", "Hi", "Hey"))
    big_rows: Final = _batch_rows(
        ("This is a longer message that will consume more tokens from the rate limit. " * 100,) * 3
    )
    _serve_file(openai_files, "file-small", small_rows)
    _serve_file(openai_files, "file-big", big_rows)
    user_api_key_dict: Final = UserAPIKeyAuth(api_key="test-key-123", tpm_limit=200, rpm_limit=10)
    _, _, small_limiter = _make_limiters()

    data_small: Final = dict(_create_batch_data("file-small"))
    result: Final = await small_limiter.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=DualCache(),
        data=data_small,
        call_type="acreate_batch",
    )

    assert result is data_small
    assert data_small["_batch_token_count"] == _token_counter_total(small_rows)
    assert data_small["_batch_request_count"] == 3

    _, _, big_limiter = _make_limiters()
    with pytest.raises(HTTPException) as exc_info:
        await big_limiter.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data=dict(_create_batch_data("file-big")),
            call_type="acreate_batch",
        )
    assert exc_info.value.status_code == 429
    assert "tokens" in str(exc_info.value.detail).lower()


@pytest.mark.asyncio
async def test_batch_rate_limit_cumulative_tpm_rejects_second_request(openai_files: respx.MockRouter):
    _, _, batch_limiter = _make_limiters()
    user_api_key_dict: Final = UserAPIKeyAuth(api_key="test-key-456", tpm_limit=200, rpm_limit=10)
    first_rows: Final = _batch_rows(("This message has some content to reach about 100 tokens total. " * 4,) * 2)
    second_rows: Final = _batch_rows(
        ("This is another message with more content to exceed the remaining limit. " * 11,) * 2
    )
    _serve_file(openai_files, "file-1", first_rows)
    _serve_file(openai_files, "file-2", second_rows)
    first_tokens: Final = _token_counter_total(first_rows)
    assert first_tokens <= 200 < first_tokens + _token_counter_total(second_rows)

    first_data: Final = dict(_create_batch_data("file-1"))
    first_result: Final = await batch_limiter.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=DualCache(),
        data=first_data,
        call_type="acreate_batch",
    )
    assert first_result is first_data
    assert first_data["_batch_token_count"] == first_tokens

    with pytest.raises(HTTPException) as exc_info:
        await batch_limiter.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=DualCache(),
            data=dict(_create_batch_data("file-2")),
            call_type="acreate_batch",
        )
    assert exc_info.value.status_code == 429
    assert "tokens" in str(exc_info.value.detail).lower()


@pytest.mark.asyncio
async def test_batch_rate_limiter_reads_a_provider_file_with_user_context(openai_files: respx.MockRouter):
    _, _, batch_limiter = _make_limiters()
    user_api_key_dict: Final = UserAPIKeyAuth(
        api_key="test-key-managed-files", user_id="test-user-abc123", tpm_limit=500, rpm_limit=10
    )
    rows: Final = _batch_rows(("This is a test message for batch rate limiting with managed files. " * 5,) * 3)
    content_route: Final = _serve_file(openai_files, "file-abc123", rows)

    data: Final = dict(_create_batch_data("file-abc123"))
    result: Final = await batch_limiter.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=DualCache(),
        data=data,
        call_type="acreate_batch",
    )

    assert content_route.call_count == 1
    assert result is data
    assert data["_batch_token_count"] == _token_counter_total(rows)
    assert data["_batch_request_count"] == 3


@pytest.mark.asyncio
async def test_batch_rate_limiter_without_user_context(openai_files: respx.MockRouter):
    _, _, batch_limiter = _make_limiters()
    rows: Final = _batch_rows(("Hello",))
    content_route: Final = _serve_file(openai_files, "file-abc123", rows)

    usage_without_context: Final = await batch_limiter.count_input_file_usage(
        file_id="file-abc123", custom_llm_provider="openai", user_api_key_dict=None
    )
    usage_with_context: Final = await batch_limiter.count_input_file_usage(
        file_id="file-abc123",
        custom_llm_provider="openai",
        user_api_key_dict=UserAPIKeyAuth(api_key="test-key", user_id="test-user-123"),
    )

    assert content_route.call_count == 2
    assert usage_without_context.request_count == usage_with_context.request_count == 1
    assert usage_without_context.total_tokens == usage_with_context.total_tokens == _token_counter_total(rows)


@pytest.mark.asyncio
async def test_managed_file_is_read_through_the_managed_files_hook_with_user_context(
    openai_files: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    from litellm_enterprise.proxy.hooks.managed_files import _PROXY_LiteLLMManagedFiles

    from litellm import Router
    from litellm.models.managed_files import LiteLLM_ManagedFileTable
    from litellm.proxy import proxy_server
    from litellm.proxy.openai_files_endpoints.common_utils import is_base64_encoded_unified_file_id
    from litellm.proxy.utils import ProxyLogging

    assert is_base64_encoded_unified_file_id(_MANAGED_FILE_ID)
    rows: Final = _batch_rows(("Test message for regression",))
    provider_route: Final = _serve_file(openai_files, "file-provider-1", rows)
    standard_route: Final = _serve_file(openai_files, "file-abc123", rows)
    unrouted_managed_read: Final = _serve_file(openai_files, _MANAGED_FILE_ID, rows)
    file_cache: Final = InternalUsageCache(dual_cache=DualCache())
    await file_cache.async_set_cache(
        key=_MANAGED_FILE_ID,
        value=LiteLLM_ManagedFileTable(
            unified_file_id=_MANAGED_FILE_ID,
            model_mappings={"deployment-1": "file-provider-1"},
            flat_model_file_ids=["file-provider-1"],
            created_by="test-user-regression",
        ).model_dump(),
        litellm_parent_otel_span=None,
    )
    proxy_logging: Final = ProxyLogging(user_api_key_cache=DualCache())
    proxy_logging.proxy_hook_mapping["managed_files"] = _PROXY_LiteLLMManagedFiles(
        internal_usage_cache=file_cache, prisma_client=None
    )
    router: Final = Router(
        model_list=[
            {
                "model_name": _BATCH_MODEL,
                "litellm_params": {"model": f"openai/{_BATCH_MODEL}", "api_key": "sk-test"},
                "model_info": {"id": "deployment-1"},
            }
        ]
    )
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", proxy_logging)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    _, _, batch_limiter = _make_limiters()
    user_api_key_dict: Final = UserAPIKeyAuth(
        api_key="test-key-regression", user_id="test-user-regression", tpm_limit=1000, rpm_limit=10
    )

    managed_usage: Final = await batch_limiter.count_input_file_usage(
        file_id=_MANAGED_FILE_ID, custom_llm_provider="openai", user_api_key_dict=user_api_key_dict
    )
    standard_usage: Final = await batch_limiter.count_input_file_usage(
        file_id="file-abc123", custom_llm_provider="openai", user_api_key_dict=user_api_key_dict
    )

    assert provider_route.call_count == 1
    assert standard_route.call_count == 1
    assert not unrouted_managed_read.called
    assert managed_usage.request_count == standard_usage.request_count == 1
    assert managed_usage.total_tokens == standard_usage.total_tokens == _token_counter_total(rows)
