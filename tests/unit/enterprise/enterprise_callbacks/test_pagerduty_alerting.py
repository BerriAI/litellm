from collections.abc import Iterator
from typing import Final, cast

import httpx
import pytest
import respx
from pydantic import TypeAdapter

import litellm
from litellm.types.utils import StandardLoggingPayload
from litellm_enterprise.enterprise_callbacks.pagerduty.pagerduty import PagerDutyAlerting
from litellm.proxy._types import UserAPIKeyAuth

PAGERDUTY_URL: Final = "https://events.pagerduty.com/v2/enqueue"


@pytest.fixture(autouse=True)
def _httpx_transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    yield
    litellm.in_memory_llm_clients_cache.flush_cache()


def _pagerduty_logger(monkeypatch: pytest.MonkeyPatch, failure_threshold: int = 1) -> PagerDutyAlerting:
    monkeypatch.setenv("PAGERDUTY_API_KEY", "unit-routing-key")
    return PagerDutyAlerting(
        alerting_args={
            "failure_threshold": failure_threshold,
            "failure_threshold_window_seconds": 600,
            "hanging_threshold_seconds": 0,
            "hanging_threshold_fails": 1,
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("failure_count", "threshold", "request_count"), [(1, 1, 1), (2, 3, 0), (3, 3, 1)])
async def test_failure_alerts_follow_the_configured_threshold(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    failure_count: int,
    threshold: int,
    request_count: int,
) -> None:
    route: Final = respx_mock.post(PAGERDUTY_URL).mock(return_value=httpx.Response(202))
    logger: Final = _pagerduty_logger(monkeypatch, threshold)
    payload: Final = cast(
        StandardLoggingPayload,
        {
            "error_information": {
                "error_class": "RateLimitError",
                "error_code": "429",
                "llm_provider": "openai",
            },
            "metadata": {},
        },
    )

    for _ in range(failure_count):
        await logger.async_log_failure_event({"standard_logging_object": payload}, None, None, None)

    assert route.call_count == request_count
    if request_count == 1:
        body: Final = TypeAdapter(dict[str, object]).validate_json(route.calls.last.request.content)
        payload_body: Final = TypeAdapter(dict[str, object]).validate_python(body["payload"])
        details: Final = TypeAdapter(dict[str, object]).validate_python(payload_body["custom_details"])
        recent_errors: Final = TypeAdapter(list[dict[str, object]]).validate_python(details["recent_errors"])
        assert body["routing_key"] == "unit-routing-key"
        assert body["event_action"] == "trigger"
        assert payload_body["summary"] == f"High LLM API Failure Rate: {failure_count} in the last 600 seconds."
        assert recent_errors[0]["error_class"] == "RateLimitError"
        assert recent_errors[0]["error_code"] == "429"
        assert recent_errors[0]["error_llm_provider"] == "openai"


@pytest.mark.asyncio
async def test_hanging_requests_alert_at_the_default_threshold(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    route: Final = respx_mock.post(PAGERDUTY_URL).mock(return_value=httpx.Response(202))
    logger: Final = _pagerduty_logger(monkeypatch)
    user: Final = UserAPIKeyAuth(api_key="unit-key", key_alias="unit-alias", user_id="unit-user")

    for _ in range(60):
        await logger.hanging_response_handler(
            {"request_id": "request-1", "litellm_call_id": "unit-call"},
            user,
        )

    assert route.call_count == 1
    body: Final = TypeAdapter(dict[str, object]).validate_json(route.calls.last.request.content)
    payload_body: Final = TypeAdapter(dict[str, object]).validate_python(body["payload"])
    details: Final = TypeAdapter(dict[str, object]).validate_python(payload_body["custom_details"])
    recent_errors: Final = TypeAdapter(list[dict[str, object]]).validate_python(details["recent_errors"])
    assert body["routing_key"] == "unit-routing-key"
    assert payload_body["summary"] == "High Number of Hanging LLM Requests: 60 in the last 600 seconds."
    assert recent_errors[0]["failure_event_type"] == "hanging_response"
