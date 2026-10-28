import asyncio
import datetime
from collections.abc import Sequence
from typing import Final, Literal, TypedDict

from typing_extensions import ReadOnly

import httpx
import pytest
import respx
from pydantic import TypeAdapter

import litellm
import litellm.proxy.proxy_server as proxy_server
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.langfuse.langfuse import LangFuseLogger, installed_langfuse_version
from litellm.integrations.langfuse.langfuse_sdk import (
    build_langfuse_client,
    build_langfuse_tracing,
    resolve_trace_id,
)
from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.integrations.SlackAlerting.utils import add_langfuse_trace_id_to_alert
from litellm.litellm_core_utils import litellm_logging
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.proxy.utils import ProxyLogging
from litellm.types.integrations.slack_alerting import AlertType
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

_WEBHOOK: Final = "https://hooks.slack.example/services/delivery"
_LANGFUSE_HOST: Final = "https://langfuse.alerts.example"
_AZURE_BASE: Final = "https://openai-gpt-4-test-v-1.openai.azure.com/"
_DAILY_BASE: Final = "https://daily-report.openai.example/v1"


class _SlackPayload(TypedDict):
    text: ReadOnly[str]


class _TeamRow(TypedDict):
    team_alias: ReadOnly[str]
    total_spend: ReadOnly[float]


class _TagRow(TypedDict):
    individual_request_tag: ReadOnly[str]
    total_spend: ReadOnly[float]


_PAYLOAD: Final = TypeAdapter(_SlackPayload)


@pytest.fixture(autouse=True)
def _slack_webhook(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("SLACK_WEBHOOK_URL", _WEBHOOK)


def _webhook(respx_mock: respx.MockRouter) -> respx.Route:
    return respx_mock.post(_WEBHOOK).mock(return_value=httpx.Response(200, text="ok"))


def _posted_texts(route: respx.Route) -> tuple[str, ...]:
    return tuple(_PAYLOAD.validate_json(call.request.content)["text"] for call in route.calls)


@pytest.mark.asyncio
async def test_slow_response_alert_names_the_azure_api_base_and_reaches_the_webhook(
    respx_mock: respx.MockRouter,
) -> None:
    route: Final = _webhook(respx_mock)
    proxy_logging: Final = ProxyLogging(user_api_key_cache=DualCache())
    proxy_logging.update_values(alerting=["slack"], alerting_threshold=100, redis_cache=None)
    start: Final = datetime.datetime(2026, 1, 1, 12, 0, 0)
    messages: Final = ({"role": "user", "content": "Hey how's it going?"},)

    helper_result: Final = proxy_logging.slack_alerting_instance._response_taking_too_long_callback_helper(
        kwargs={
            "model": "chatgpt-v-3",
            "messages": messages,
            "litellm_params": {"api_base": _AZURE_BASE, "custom_llm_provider": "azure"},
        },
        start_time=start,
        end_time=start + datetime.timedelta(seconds=150),
    )

    assert helper_result == (150.0, "chatgpt-v-3", _AZURE_BASE, str(messages)[:100])

    slow_message: Final = (
        f"`Responses are slow - 150.0s response time > Alerting threshold: 100s`\nAPI Base: `{_AZURE_BASE}`"
    )
    await proxy_logging.alerting_handler(message=slow_message, level="Low", alert_type=AlertType.llm_too_slow)
    await proxy_logging.slack_alerting_instance.flush_queue()

    texts: Final = _posted_texts(route)
    assert len(texts) == 1
    assert texts[0].startswith("Alert type: `llm_too_slow`\nLevel: `Low`\n")
    assert texts[0].endswith(f"Message: {slow_message}")


@pytest.mark.asyncio
async def test_send_alert_is_queued_until_flush_then_posted_to_the_webhook_once(respx_mock: respx.MockRouter) -> None:
    route: Final = _webhook(respx_mock)
    slack_alerting: Final = SlackAlerting(alerting_threshold=1, internal_usage_cache=DualCache(), alerting=["slack"])

    await slack_alerting.send_alert("Test message", "Low", AlertType.budget_alerts, alerting_metadata={})

    assert route.call_count == 0

    await slack_alerting.flush_queue()
    await slack_alerting.flush_queue()

    texts: Final = _posted_texts(route)
    assert len(texts) == 1
    assert texts[0].startswith("Alert type: `budget_alerts`\nLevel: `Low`\n")
    assert texts[0].endswith("Message: Test message")


@pytest.mark.asyncio
async def test_a_queued_alert_is_posted_by_the_periodic_flush_without_a_manual_flush(
    respx_mock: respx.MockRouter,
) -> None:
    delivered: Final = asyncio.Event()

    def deliver(request: httpx.Request) -> httpx.Response:
        delivered.set()
        return httpx.Response(200, text="ok")

    route: Final = respx_mock.post(_WEBHOOK).mock(side_effect=deliver)
    slack_alerting: Final = SlackAlerting(alerting_threshold=1, internal_usage_cache=DualCache(), alerting=["slack"])
    slack_alerting.flush_interval = 0
    slack_alerting.update_values(alerting=["slack"])
    flush_task: Final = slack_alerting._periodic_flush_task
    assert flush_task is not None
    try:
        await slack_alerting.send_alert("Timed message", "Low", AlertType.budget_alerts, alerting_metadata={})
        await asyncio.wait_for(delivered.wait(), timeout=5)
    finally:
        flush_task.cancel()

    texts: Final = _posted_texts(route)
    assert len(texts) == 1
    assert texts[0].endswith("Message: Timed message")


class _DeploymentSettled(CustomLogger):
    def __init__(self, model_id: str) -> None:
        super().__init__()
        self.model_id: Final = model_id
        self.succeeded: Final = asyncio.Event()
        self.failed: Final = asyncio.Event()

    def _is_mine(self, kwargs: dict[str, object]) -> bool:
        payload: Final = kwargs.get("standard_logging_object")
        return isinstance(payload, dict) and payload.get("model_id") == self.model_id

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        if self._is_mine(kwargs):
            self.succeeded.set()

    async def async_log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        if self._is_mine(kwargs):
            self.failed.set()


@pytest.mark.asyncio
async def test_daily_report_lists_router_latency_after_success_and_failures_after_an_auth_error(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    webhook: Final = _webhook(respx_mock)
    respx_mock.post(f"{_DAILY_BASE}/chat/completions").mock(
        side_effect=(
            httpx.Response(
                200,
                json={
                    "id": "chatcmpl-daily",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "gpt-5-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "fine"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
                },
            ),
            httpx.Response(
                401,
                json={
                    "error": {
                        "message": "Incorrect API key provided",
                        "type": "invalid_request_error",
                        "code": "invalid_api_key",
                    }
                },
            ),
        )
    )
    model_id: Final = "daily-report-deployment"
    slack_alerting: Final = SlackAlerting(
        alerting=["slack"], internal_usage_cache=DualCache(), alert_types=[AlertType.daily_reports]
    )
    settled: Final = _DeploymentSettled(model_id)
    monkeypatch.setattr(litellm, "callbacks", [slack_alerting, settled])
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "daily-report-model",
                "litellm_params": {"model": "openai/gpt-5-mini", "api_key": "sk-daily", "api_base": _DAILY_BASE},
                "model_info": {"id": model_id},
            }
        ]
    )
    request: Final = ({"role": "user", "content": "Hey, how's it going?"},)

    await router.acompletion(model="daily-report-model", messages=list(request))
    await asyncio.wait_for(settled.succeeded.wait(), timeout=5)
    after_success: Final = await slack_alerting.send_daily_reports(router=router)
    await slack_alerting.flush_queue()

    with pytest.raises(litellm.AuthenticationError):
        await router.acompletion(model="daily-report-model", messages=list(request))
    await asyncio.wait_for(settled.failed.wait(), timeout=5)
    after_failure: Final = await slack_alerting.send_daily_reports(router=router)
    await slack_alerting.flush_queue()

    texts: Final = _posted_texts(webhook)
    assert (after_success, after_failure) == (True, True)
    assert len(texts) == 2
    assert "Most Failed Requests:*\n\n\tNone\n" in texts[0]
    assert "1. Deployment: `openai/gpt-5-mini`, Latency per output token: `" in texts[0]
    assert f"1. Deployment: `openai/gpt-5-mini`, Failed Requests: `1`,  API Base: `{_DAILY_BASE}`" in texts[1]
    assert "Top Slowest Deployments:*\n\n\tNone\n" in texts[1]


class _CallLogged(CustomLogger):
    def __init__(self, call_id: str, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__()
        self.call_id: Final = call_id
        self.loop: Final = loop
        self.logged: Final = asyncio.Event()

    def log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        if kwargs.get("litellm_call_id") == self.call_id:
            self.loop.call_soon_threadsafe(self.logged.set)


@pytest.mark.asyncio
async def test_langfuse_trace_link_ends_with_the_trace_id_the_logger_emitted(monkeypatch: pytest.MonkeyPatch) -> None:
    logger: Final = LangFuseLogger.__new__(LangFuseLogger)
    logger.tracing = build_langfuse_tracing(
        exporter=InMemorySpanExporter(), environment=None, release=None, sample_rate=1.0, flush_interval_millis=10
    )
    logger.api_client = build_langfuse_client(
        public_key="pk-alert-trace", secret_key="sk-alert-trace", base_url=_LANGFUSE_HOST, httpx_client=None
    )
    logger.langfuse_sdk_version = installed_langfuse_version()
    call_id: Final = "slack-alert-langfuse-trace"
    logged: Final = _CallLogged(call_id, asyncio.get_running_loop())
    monkeypatch.setenv("LANGFUSE_HOST", _LANGFUSE_HOST)
    monkeypatch.setattr(litellm_logging, "langFuseLogger", logger)
    monkeypatch.setattr(litellm, "success_callback", ["langfuse", logged])
    monkeypatch.setattr(litellm, "_async_success_callback", [])
    monkeypatch.setattr(litellm, "callbacks", [])
    logging_obj: Final = Logging(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="completion",
        litellm_call_id=call_id,
        start_time=datetime.datetime.now(),
        function_id=call_id,
    )

    litellm.completion(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "Hey how's it going?"}],
        mock_response="Hey!",
        litellm_logging_obj=logging_obj,
    )
    await asyncio.wait_for(logged.logged.wait(), timeout=5)
    trace_url: Final = await add_langfuse_trace_id_to_alert(request_data={"litellm_logging_obj": logging_obj})

    expected_trace_id: Final = resolve_trace_id(logging_obj.litellm_trace_id)
    assert logging_obj.get_trace_id(service_name="langfuse") == expected_trace_id
    assert trace_url == f"{_LANGFUSE_HOST}/trace/{expected_trace_id}"


class _SpendReportDb:
    def __init__(self, teams: Sequence[_TeamRow], tags: Sequence[_TagRow]) -> None:
        self.teams: Final = teams
        self.tags: Final = tags

    async def query_raw(self, query: str, *args: object) -> Sequence[_TeamRow] | Sequence[_TagRow]:
        return self.teams if "team_alias" in query else self.tags


class _SpendReportPrisma:
    def __init__(self, db: _SpendReportDb) -> None:
        self.db: Final = db


@pytest.mark.parametrize("report_type", ["weekly", "monthly"])
@pytest.mark.asyncio
async def test_spend_report_is_sent_once_per_period(
    report_type: Literal["weekly", "monthly"], respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    route: Final = _webhook(respx_mock)
    monkeypatch.setattr(
        proxy_server,
        "prisma_client",
        _SpendReportPrisma(
            _SpendReportDb(
                teams=(
                    _TeamRow(team_alias="team1", total_spend=100.0),
                    _TeamRow(team_alias="team2", total_spend=200.0),
                ),
                tags=(
                    _TagRow(individual_request_tag="tag1", total_spend=150.0),
                    _TagRow(individual_request_tag="tag2", total_spend=150.0),
                ),
            )
        ),
    )
    slack_alerting: Final = SlackAlerting(alerting=["slack"], internal_usage_cache=DualCache())
    send_report: Final = (
        slack_alerting.send_weekly_spend_report if report_type == "weekly" else slack_alerting.send_monthly_spend_report
    )

    await send_report()
    await slack_alerting.flush_queue()
    await send_report()
    await slack_alerting.flush_queue()

    texts: Final = _posted_texts(route)
    assert len(texts) == 1
    assert "Team: `team1` | Spend: `$100.0`\nTeam: `team2` | Spend: `$200.0`\n" in texts[0]
    assert "Tag: `tag1` | Spend: `$150.0`\nTag: `tag2` | Spend: `$150.0`\n" in texts[0]
