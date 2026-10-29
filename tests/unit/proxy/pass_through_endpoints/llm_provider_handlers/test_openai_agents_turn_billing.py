import asyncio
import json
import logging
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy._types import SpendLogsPayload
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_agents_turn_billing import (
    OpenAIAgentsTurnBiller,
    live_agents_api_get,
)
from litellm.proxy.pass_through_endpoints.streaming_handler import PassThroughStreamingHandler
from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    EndpointType,
    PassthroughStandardLoggingPayload,
)

SESSIONS_URL: Final = "https://api.openai.com/v1/agents/sessions"
SESSION_ID: Final = "sess_6a0b0f8e2c1d4e5f"
EVENTS_URL: Final = f"{SESSIONS_URL}/{SESSION_ID}/events"
TURNS_URL: Final = f"{SESSIONS_URL}/{SESSION_ID}/turns?order=desc&limit=100"
CALL_TIME: Final = datetime(2026, 10, 29, 9, 12, 41)
CALL_STARTED_AT: Final = math.floor(CALL_TIME.timestamp())
AGENT_MODEL: Final = "gpt-5.4-mini"
TURN_USAGE: Final = {
    "input_tokens": 7200,
    "input_tokens_details": {"cached_tokens": 6912},
    "output_tokens": 40,
    "output_tokens_details": {"reasoning_tokens": 16},
    "total_tokens": 7240,
}
SECOND_TURN_USAGE: Final = {
    "input_tokens": 7310,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens": 95,
    "output_tokens_details": {"reasoning_tokens": 64},
    "total_tokens": 7405,
}
INPUT_MESSAGE: Final = {"type": "agent.session.input.message", "content": [{"type": "input_text", "text": "Say hi"}]}
TOOL_RESULT: Final = {"type": "agent.session.input.tool_result", "call_id": "call_5c1d", "output": "42"}


def _agent(service_tier: str) -> dict[str, object]:
    return {"id": "agent_6a0b0f8e", "model": AGENT_MODEL, "service_tier": service_tier, "tools": []}


def _session(service_tier: str = "auto") -> dict[str, object]:
    return {
        "id": SESSION_ID,
        "object": "agent.session",
        "created_at": CALL_STARTED_AT,
        "status": "in_progress",
        "agent": _agent(service_tier),
        "usage": None,
    }


def _turn(turn_id: str, created_at: int, status: str, usage: Mapping[str, object] | None) -> dict[str, object]:
    return {"id": turn_id, "object": "agent.session.turn", "created_at": created_at, "status": status, "usage": usage}


def _expected_cost(usage: Mapping[str, object], service_tier: str = "auto") -> float:
    rates: Final = litellm.model_cost[AGENT_MODEL]
    suffix: Final = "" if service_tier == "auto" else f"_{service_tier}"
    input_details: Final = usage["input_tokens_details"]
    assert isinstance(input_details, dict)
    cached: Final = input_details["cached_tokens"]
    return (
        (usage["input_tokens"] - cached) * rates[f"input_cost_per_token{suffix}"]
        + cached * rates[f"cache_read_input_token_cost{suffix}"]
        + usage["output_tokens"] * rates[f"output_cost_per_token{suffix}"]
    )


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds
        await asyncio.sleep(0)


class _AgentsApi:
    def __init__(
        self,
        turn_pages: Sequence[Sequence[Mapping[str, object]]],
        session: Mapping[str, object] | None = None,
        status_code: int = 200,
        unreachable: bool = False,
    ) -> None:
        self._turn_pages: Final = turn_pages
        self._session: Final = session
        self._status_code: Final = status_code
        self._unreachable: Final = unreachable
        self.urls: Final[list[str]] = []
        self.turn_polls = 0

    async def get(self, base_url: str, path: str) -> httpx.Response:
        url: Final = f"{base_url}/{path}"
        self.urls.append(url)
        request: Final = httpx.Request("GET", url)
        if self._unreachable:
            raise httpx.ConnectError("connection refused", request=request)
        if self._status_code != 200:
            return httpx.Response(self._status_code, json={"error": {"message": "upstream"}}, request=request)
        if url == TURNS_URL:
            page: Final = self._turn_pages[min(self.turn_polls, len(self._turn_pages) - 1)]
            self.turn_polls += 1
            return httpx.Response(200, json={"object": "list", "data": list(page)}, request=request)
        if url == f"{SESSIONS_URL}/{SESSION_ID}" and self._session is not None:
            return httpx.Response(200, json=dict(self._session), request=request)
        return httpx.Response(404, json={"error": {"message": "not found"}}, request=request)


class _SpendRows(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.rows: Final[list[SpendLogsPayload]] = []
        self.logged: Final = asyncio.Event()

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.rows.append(
            get_logging_payload(kwargs=kwargs, response_obj=response_obj, start_time=start_time, end_time=end_time)
        )
        self.logged.set()

    async def only_row(self) -> SpendLogsPayload:
        await asyncio.wait_for(self.logged.wait(), timeout=5)
        for _ in range(50):
            await asyncio.sleep(0)
        assert len(self.rows) == 1
        return self.rows[0]


def _logging_obj(rows: _SpendRows, litellm_call_id: str, stream: bool) -> LiteLLMLoggingObj:
    logging_obj: Final = LiteLLMLoggingObj(
        model="unknown",
        messages=[],
        stream=stream,
        call_type="pass_through_endpoint",
        start_time=CALL_TIME,
        litellm_call_id=litellm_call_id,
        function_id="1234",
        dynamic_async_success_callbacks=[rows],
    )
    logging_obj.update_environment_variables(
        model="unknown", user="unknown", optional_params={}, litellm_params={}, call_type="pass_through_endpoint"
    )
    logging_obj.model_call_details["custom_llm_provider"] = "openai"
    return logging_obj


def _biller(api: _AgentsApi, clock: _Clock) -> OpenAIAgentsTurnBiller:
    return OpenAIAgentsTurnBiller(get=api.get, sleep=clock.sleep, clock=clock)


async def _relay_call(
    biller: OpenAIAgentsTurnBiller,
    rows: _SpendRows,
    url_route: str,
    request_body: dict[str, object],
    response_body: dict[str, object] | None,
    litellm_call_id: str,
    status_code: int = 200,
) -> None:
    await PassThroughEndpointLogging(agents_turn_biller=biller).pass_through_async_success_handler(
        httpx_response=httpx.Response(status_code, json=response_body, request=httpx.Request("POST", url_route)),
        response_body=response_body,
        logging_obj=_logging_obj(rows, litellm_call_id, stream=False),
        url_route=url_route,
        result="",
        start_time=CALL_TIME,
        end_time=CALL_TIME,
        cache_hit=False,
        request_body=request_body,
        passthrough_logging_payload=PassthroughStandardLoggingPayload(
            url=url_route, request_body=request_body, request_method="POST"
        ),
        custom_llm_provider="openai",
    )


@pytest.mark.asyncio
async def test_an_input_message_bills_its_turn_once_usage_arrives(local_model_cost_map: None):
    api: Final = _AgentsApi(
        turn_pages=(
            (),
            (_turn("turn_1", CALL_STARTED_AT + 1, "in_progress", SECOND_TURN_USAGE),),
            (_turn("turn_1", CALL_STARTED_AT + 1, "completed", None),),
            (_turn("turn_1", CALL_STARTED_AT + 1, "completed", TURN_USAGE),),
        ),
        session=_session(),
    )
    rows: Final = _SpendRows()

    await _relay_call(
        _biller(api, _Clock()), rows, EVENTS_URL, {"events": [INPUT_MESSAGE]}, None, "9b1d3f5a-7c9e-4b2d-8f4a-6c8e0a2b4d61"
    )
    row: Final = await rows.only_row()

    assert api.turn_polls == 4
    assert TURNS_URL in api.urls
    assert row["model"] == AGENT_MODEL
    assert row["prompt_tokens"] == TURN_USAGE["input_tokens"]
    assert row["completion_tokens"] == TURN_USAGE["output_tokens"]
    assert row["spend"] > 0
    assert row["spend"] == pytest.approx(_expected_cost(TURN_USAGE))


@pytest.mark.asyncio
@pytest.mark.parametrize("service_tier", ["auto", "priority", "flex"])
async def test_a_session_create_with_input_bills_its_first_turn_at_the_agent_service_tier(
    local_model_cost_map: None, service_tier: str
):
    api: Final = _AgentsApi(turn_pages=((_turn("turn_1", CALL_STARTED_AT, "completed", TURN_USAGE),),))
    rows: Final = _SpendRows()

    await _relay_call(
        _biller(api, _Clock()),
        rows,
        SESSIONS_URL,
        {"agent": {"model": AGENT_MODEL}, "input": "Say hi in one word"},
        _session(service_tier),
        "1e3a5c7e-9b2d-4f6a-8c0e-2a4c6e8a0b17",
    )
    row: Final = await rows.only_row()

    assert row["spend"] == pytest.approx(_expected_cost(TURN_USAGE, service_tier))
    assert api.urls == [TURNS_URL]


@pytest.mark.asyncio
async def test_two_calls_on_one_session_bill_two_different_turns(local_model_cost_map: None):
    api: Final = _AgentsApi(
        turn_pages=(
            (
                _turn("turn_2", CALL_STARTED_AT, "completed", SECOND_TURN_USAGE),
                _turn("turn_1", CALL_STARTED_AT, "completed", TURN_USAGE),
            ),
        ),
        session=_session(),
    )
    biller: Final = _biller(api, _Clock())
    first: Final = _SpendRows()
    second: Final = _SpendRows()

    await _relay_call(biller, first, EVENTS_URL, {"events": [INPUT_MESSAGE]}, None, "3c5e7a9c-1d3f-4a6b-9e2c-4b6d8f0a2c38")
    await _relay_call(biller, second, EVENTS_URL, {"events": [INPUT_MESSAGE]}, None, "5e7a9c1e-3f5b-4c8d-ae4c-6d8f0b2c4e59")
    billed: Final = sorted([(await first.only_row())["total_tokens"], (await second.only_row())["total_tokens"]])

    assert billed == sorted([TURN_USAGE["total_tokens"], SECOND_TURN_USAGE["total_tokens"]])


@pytest.mark.asyncio
async def test_a_turn_created_before_the_call_is_never_billed_to_it(local_model_cost_map: None):
    api: Final = _AgentsApi(
        turn_pages=(
            (
                _turn("turn_new", CALL_STARTED_AT + 2, "completed", TURN_USAGE),
                _turn("turn_old", CALL_STARTED_AT - 1, "completed", SECOND_TURN_USAGE),
            ),
        ),
        session=_session(),
    )
    rows: Final = _SpendRows()

    await _relay_call(
        _biller(api, _Clock()), rows, EVENTS_URL, {"events": [INPUT_MESSAGE]}, None, "7a9c1e3a-5b7d-4e0f-8a6e-8f0b2d4e6a70"
    )

    assert (await rows.only_row())["total_tokens"] == TURN_USAGE["total_tokens"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("turn_pages", "status_code", "unreachable"),
    [
        pytest.param(((_turn("turn_1", CALL_STARTED_AT + 1, "in_progress", None),),), 200, False, id="turn_never_ends"),
        pytest.param(((_turn("turn_1", CALL_STARTED_AT + 1, "completed", None),),), 200, False, id="usage_never_arrives"),
        pytest.param(
            ((_turn("turn_old", CALL_STARTED_AT - 1, "completed", TURN_USAGE),),), 200, False, id="no_turn_to_claim"
        ),
        pytest.param((({"id": "turn_1", "status": "completed"},),), 200, False, id="turns_page_unreadable"),
        pytest.param(((),), 500, False, id="turns_list_fails"),
        pytest.param(((),), 200, True, id="turns_list_unreachable"),
    ],
)
async def test_a_turn_that_never_reports_usage_logs_the_call_once_at_zero(
    local_model_cost_map: None,
    turn_pages: Sequence[Sequence[Mapping[str, object]]],
    status_code: int,
    unreachable: bool,
):
    rows: Final = _SpendRows()

    await _relay_call(
        _biller(
            _AgentsApi(turn_pages=turn_pages, session=_session(), status_code=status_code, unreachable=unreachable),
            _Clock(),
        ),
        rows,
        EVENTS_URL,
        {"events": [INPUT_MESSAGE]},
        None,
        "9c1e3a5c-7d9f-4a2b-8c0a-0b2d4f6a8c91",
    )
    row: Final = await rows.only_row()

    assert not row["spend"]
    assert row["total_tokens"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "session",
    [
        pytest.param(None, id="session_not_found"),
        pytest.param({"id": SESSION_ID, "agent": "not-an-agent"}, id="session_unreadable"),
        pytest.param({"id": SESSION_ID, "agent": {"model": "agents-model-without-a-price"}}, id="unpriced_model"),
    ],
)
async def test_an_unknown_agent_model_still_logs_the_turn_tokens(
    local_model_cost_map: None, session: Mapping[str, object] | None
):
    rows: Final = _SpendRows()
    turns: Final = ((_turn("turn_1", CALL_STARTED_AT, "completed", TURN_USAGE),),)

    await _relay_call(
        _biller(_AgentsApi(turn_pages=turns, session=session), _Clock()),
        rows,
        EVENTS_URL,
        {"events": [INPUT_MESSAGE]},
        None,
        "2d4f6a8c-0e2a-4b4c-9d6e-1c3e5a7c9e02",
    )
    row: Final = await rows.only_row()

    assert not row["spend"]
    assert row["total_tokens"] == TURN_USAGE["total_tokens"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url_route", "request_body", "response_body", "status_code"),
    [
        pytest.param(EVENTS_URL, {"events": [TOOL_RESULT]}, None, 202, id="tool_result"),
        pytest.param(EVENTS_URL, {"events": [{"type": "agent.session.input.cancel"}]}, None, 202, id="cancel"),
        pytest.param(EVENTS_URL, {"events": "agent.session.input.message"}, None, 202, id="malformed_events"),
        pytest.param(SESSIONS_URL, {"input": 42}, _session(), 201, id="malformed_input"),
        pytest.param(SESSIONS_URL, {"agent": {"model": AGENT_MODEL}}, _session(), 201, id="create_without_input"),
        pytest.param(SESSIONS_URL, {"input": "Say hi"}, {"id": "../files"}, 201, id="unsafe_session_id"),
        pytest.param(f"{SESSIONS_URL}/{SESSION_ID}/turns", {"input": "Say hi"}, None, 200, id="other_agents_route"),
    ],
)
async def test_a_call_that_starts_no_turn_logs_at_once_without_polling(
    local_model_cost_map: None,
    url_route: str,
    request_body: dict[str, object],
    response_body: dict[str, object] | None,
    status_code: int,
):
    api: Final = _AgentsApi(turn_pages=((_turn("turn_1", CALL_STARTED_AT, "completed", TURN_USAGE),),))
    rows: Final = _SpendRows()

    await _relay_call(
        _biller(api, _Clock()),
        rows,
        url_route,
        request_body,
        response_body,
        "4f6a8c0e-2a4c-4d6e-8f0a-3e5a7c9e1a24",
        status_code,
    )
    row: Final = await rows.only_row()

    assert api.urls == []
    assert not row["spend"]


@pytest.mark.asyncio
async def test_a_streamed_session_create_defers_its_row_until_the_turn_is_billed(local_model_cost_map: None):
    api: Final = _AgentsApi(turn_pages=((_turn("turn_1", CALL_STARTED_AT, "completed", TURN_USAGE),),))
    rows: Final = _SpendRows()
    created_event: Final = {"type": "agent.session.created", "session": _session()}

    await PassThroughStreamingHandler.route_streaming_logging_to_handler(
        litellm_logging_obj=_logging_obj(rows, "6a8c0e2a-4c6e-4f8a-9b2c-5a7c9e1a3c46", stream=True),
        passthrough_success_handler_obj=PassThroughEndpointLogging(agents_turn_biller=_biller(api, _Clock())),
        url_route=SESSIONS_URL,
        request_body={"agent": {"model": AGENT_MODEL}, "input": "Say hi in one word", "stream": True},
        endpoint_type=EndpointType.OPENAI,
        start_time=CALL_TIME,
        raw_bytes=(
            b": keep-alive\n\nevent: agent.session.queued\ndata: {\"type\": \"agent.session.queued\"}\n\n",
            f"event: agent.session.created\ndata: {json.dumps(created_event)}\n\n".encode(),
            b'event: agent.turn.completed\ndata: {"type": "agent.turn.completed", "usage": null}\n\n',
        ),
        end_time=CALL_TIME,
    )
    row: Final = await rows.only_row()

    assert api.urls == [TURNS_URL]
    assert row["model"] == AGENT_MODEL
    assert row["total_tokens"] == TURN_USAGE["total_tokens"]
    assert row["spend"] == pytest.approx(_expected_cost(TURN_USAGE))


@pytest.mark.asyncio
async def test_the_turns_poll_sends_the_passthrough_openai_key_and_the_agents_beta_header(
    local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-agents-turns-poll")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    litellm.in_memory_llm_clients_cache.flush_cache()
    clock: Final = _Clock()
    rows: Final = _SpendRows()
    turns_page: Final = {"object": "list", "data": [_turn("turn_1", CALL_STARTED_AT, "completed", TURN_USAGE)]}

    with respx.mock(assert_all_called=True) as upstream:
        turns: Final = upstream.get(TURNS_URL).respond(json=turns_page)
        await _relay_call(
            OpenAIAgentsTurnBiller(get=live_agents_api_get, sleep=clock.sleep, clock=clock),
            rows,
            SESSIONS_URL,
            {"agent": {"model": AGENT_MODEL}, "input": "Say hi in one word"},
            _session(),
            "8c0e2a4c-6e8a-4b0c-9d4e-7c9e1a3c5e68",
        )
        row: Final = await rows.only_row()
    litellm.in_memory_llm_clients_cache.flush_cache()

    assert turns.calls.last.request.headers["authorization"] == "Bearer sk-agents-turns-poll"
    assert turns.calls.last.request.headers["openai-beta"] == "agents=v1"
    assert row["spend"] == pytest.approx(_expected_cost(TURN_USAGE))


async def _cancelled_sleep(seconds: float) -> None:
    raise asyncio.CancelledError


async def _failing_sleep(seconds: float) -> None:
    raise RuntimeError("event loop is closing")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sleep", "lost_reason"),
    [pytest.param(_cancelled_sleep, "cancelled", id="cancelled"), pytest.param(_failing_sleep, "failed", id="failed")],
)
async def test_a_billing_task_that_dies_reports_the_lost_row(
    local_model_cost_map: None,
    caplog: pytest.LogCaptureFixture,
    sleep: Callable[[float], Awaitable[None]],
    lost_reason: str,
):
    api: Final = _AgentsApi(turn_pages=((_turn("turn_1", CALL_STARTED_AT, "completed", TURN_USAGE),),))
    rows: Final = _SpendRows()

    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        await _relay_call(
            OpenAIAgentsTurnBiller(get=api.get, sleep=sleep, clock=_Clock()),
            rows,
            EVENTS_URL,
            {"events": [INPUT_MESSAGE]},
            None,
            "0e2a4c6e-8a0c-4e2d-8b6c-9e1a3c5e7a80",
        )
        for _ in range(50):
            await asyncio.sleep(0)

    assert rows.rows == []
    assert any(f"turn billing {lost_reason}, the call logged no row" in message for message in caplog.messages)
