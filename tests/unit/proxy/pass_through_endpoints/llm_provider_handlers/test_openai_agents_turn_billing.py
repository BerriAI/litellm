import asyncio
import contextvars
import json
import logging
import math
import re
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Final, TypeAlias

import httpx
import openai
import pytest
import respx

import litellm
from litellm.caching.caching import DualCache
from litellm.constants import OPENAI_AGENTS_SETTLED_FETCH_ATTEMPTS, OPENAI_AGENTS_TURN_CLOCK_SKEW_SECONDS
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.openai.common_utils import OpenAIError as LiteLLMOpenAIError
from litellm.proxy._types import SpendLogsPayload, UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.parallel_request_limiter import PROXY_MaxParallelRequestsHandler
from litellm.proxy.hooks.parallel_request_limiter_v3 import PROXY_MaxParallelRequestsHandler_v3
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.openai_agents_turn_billing import (
    OpenAIAgentsTurnBiller,
    TurnClaims,
    live_agents_api_get,
)
from litellm.proxy.pass_through_endpoints.streaming_handler import PassThroughStreamingHandler
from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload
from litellm.proxy.utils import InternalUsageCache, ProxyLogging, hash_token
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    EndpointType,
    PassthroughStandardLoggingPayload,
)

SESSIONS_URL: Final = "https://api.openai.com/v1/agents/sessions"
SESSION_ID: Final = "sess_6a0b0f8e2c1d4e5f"
EVENTS_URL: Final = f"{SESSIONS_URL}/{SESSION_ID}/events"
TURNS_URL: Final = f"{SESSIONS_URL}/{SESSION_ID}/turns?order=desc&limit=100"
SUBAGENTS_URL: Final = f"{SESSIONS_URL}/{SESSION_ID}/subagents?order=desc&limit=100"
SUBAGENT_TURNS_URL: Final = f"{SESSIONS_URL}/{SESSION_ID}/subagents/subagent_1/turns?order=desc&limit=100"
ROOT_ITEMS_URL: Final = f"{SESSIONS_URL}/{SESSION_ID}/turns/turn_1/items?limit=100"
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
USER_TEXT: Final = "Say hi"
TOOL_RESULT: Final = {"type": "agent.session.input.tool_result", "call_id": "call_5c1d", "output": "42"}
TURN_ITEMS_PATH: Final = re.compile(r"/turns/(?P<turn_id>[^/]+)/items\?order=asc&limit=100$")


def _input_message(text: str) -> dict[str, object]:
    return {
        "type": "agent.session.input.message",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": text}]}],
    }


INPUT_MESSAGE: Final = _input_message(USER_TEXT)


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


def _page(*data: Mapping[str, object]) -> dict[str, object]:
    return {"object": "list", "data": list(data)}


def _user_messages(texts: Sequence[str]) -> dict[str, object]:
    return _page(
        *({"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]} for text in texts)
    )


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


_Reply: TypeAlias = Mapping[str, object] | int | Exception


class _AgentsApi:
    def __init__(
        self,
        turn_pages: Sequence[Sequence[Mapping[str, object]]],
        session: Mapping[str, object] | None = None,
        status_code: int = 200,
        unreachable: bool = False,
        replies: Mapping[str, Sequence[_Reply]] | None = None,
        turn_texts: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self._turn_pages: Final = turn_pages
        self._turn_texts: Final = turn_texts or {}
        self._session: Final = session
        self._status_code: Final = status_code
        self._unreachable: Final = unreachable
        self._replies: Final = replies or {}
        self.urls: Final[list[str]] = []
        self.turn_polls = 0

    def _reply(self, url: str, request: httpx.Request) -> httpx.Response:
        replies: Final = self._replies[url]
        reply: Final = replies[min(self.urls.count(url), len(replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, int):
            return httpx.Response(reply, json={"error": {"message": "upstream"}}, request=request)
        return httpx.Response(200, json=dict(reply), request=request)

    async def get(self, base_url: str, path: str, scope_headers: Mapping[str, str]) -> httpx.Response:
        url: Final = f"{base_url}/{path}"
        self.urls.append(url)
        request: Final = httpx.Request("GET", url)
        if self._unreachable:
            raise httpx.ConnectError("connection refused", request=request)
        if self._status_code != 200:
            return httpx.Response(self._status_code, json={"error": {"message": "upstream"}}, request=request)
        if url in self._replies:
            return self._reply(url, request)
        if url == TURNS_URL:
            page: Final = self._turn_pages[min(self.turn_polls, len(self._turn_pages) - 1)]
            self.turn_polls += 1
            return httpx.Response(200, json={"object": "list", "data": list(page)}, request=request)
        if url == f"{SESSIONS_URL}/{SESSION_ID}" and self._session is not None:
            return httpx.Response(200, json=dict(self._session), request=request)
        turn_items: Final = TURN_ITEMS_PATH.search(url)
        if turn_items is not None:
            texts: Final = self._turn_texts.get(turn_items["turn_id"], (USER_TEXT,))
            return httpx.Response(200, json=_user_messages(texts), request=request)
        if url == SUBAGENTS_URL or "/items?" in url:
            return httpx.Response(200, json=_page(), request=request)
        return httpx.Response(404, json={"error": {"message": "not found"}}, request=request)


class _SharedRedis:
    def __init__(self) -> None:
        self.values: Final[dict[str, object]] = {}

    async def async_set_cache(self, key: str, value: object, *, nx: bool, ttl: int) -> object:
        if nx and key in self.values:
            return None
        self.values[key] = value
        return True

    async def async_get_cache(self, key: str) -> object:
        return self.values.get(key)


class _DownRedis:
    async def async_set_cache(self, key: str, value: object, *, nx: bool, ttl: int) -> object:
        raise ConnectionError("redis is down")

    async def async_get_cache(self, key: str) -> object:
        raise ConnectionError("redis is down")


class _SwallowingRedis:
    async def async_set_cache(self, key: str, value: object, *, nx: bool, ttl: int) -> object:
        return None

    async def async_get_cache(self, key: str) -> object:
        return None


class _SlowDownRedis:
    async def async_set_cache(self, key: str, value: object, *, nx: bool, ttl: int) -> object:
        await asyncio.sleep(0)
        raise ConnectionError("redis is down")

    async def async_get_cache(self, key: str) -> object:
        await asyncio.sleep(0)
        raise ConnectionError("redis is down")


class _SpendRows(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.rows: Final[list[SpendLogsPayload]] = []
        self.standard_logging_objects: Final[list[object]] = []
        self.logged: Final = asyncio.Event()

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        self.rows.append(
            get_logging_payload(kwargs=kwargs, response_obj=response_obj, start_time=start_time, end_time=end_time)
        )
        self.standard_logging_objects.append(kwargs.get("standard_logging_object"))
        self.logged.set()

    async def only_row(self) -> SpendLogsPayload:
        await asyncio.wait_for(self.logged.wait(), timeout=5)
        for _ in range(50):
            await asyncio.sleep(0)
        assert len(self.rows) == 1
        return self.rows[0]


def _litellm_params(
    request_headers: Mapping[str, str] | None = None, metadata: Mapping[str, object] | None = None
) -> dict[str, object]:
    return {"metadata": dict(metadata or {}), "proxy_server_request": {"headers": dict(request_headers or {})}}


def _logging_obj(
    rows: _SpendRows,
    litellm_call_id: str,
    stream: bool,
    api_base: str,
    litellm_params: dict[str, object],
    callbacks: Sequence[CustomLogger] = (),
) -> LiteLLMLoggingObj:
    logging_obj: Final = LiteLLMLoggingObj(
        model="unknown",
        messages=[],
        stream=stream,
        call_type="pass_through_endpoint",
        start_time=CALL_TIME,
        litellm_call_id=litellm_call_id,
        function_id="1234",
        dynamic_async_success_callbacks=[rows, *callbacks],
    )
    logging_obj.update_environment_variables(
        model="unknown",
        user="unknown",
        optional_params={},
        litellm_params=litellm_params,
        call_type="pass_through_endpoint",
    )
    logging_obj.pre_call(input=[], api_key="", additional_args={"api_base": api_base})
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
    request_headers: Mapping[str, str] | None = None,
    metadata: Mapping[str, object] | None = None,
    callbacks: Sequence[CustomLogger] = (),
) -> None:
    litellm_params: Final = _litellm_params(request_headers, metadata)
    await PassThroughEndpointLogging(agents_turn_biller=biller).pass_through_async_success_handler(
        httpx_response=httpx.Response(status_code, json=response_body, request=httpx.Request("POST", url_route)),
        response_body=response_body,
        logging_obj=_logging_obj(
            rows, litellm_call_id, stream=False, api_base=url_route, litellm_params=litellm_params, callbacks=callbacks
        ),
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
        litellm_params=litellm_params,
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
        _biller(api, _Clock()),
        rows,
        EVENTS_URL,
        {"events": [INPUT_MESSAGE]},
        None,
        "9b1d3f5a-7c9e-4b2d-8f4a-6c8e0a2b4d61",
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
        {"agent": {"model": AGENT_MODEL}, "input": USER_TEXT},
        _session(service_tier),
        "1e3a5c7e-9b2d-4f6a-8c0e-2a4c6e8a0b17",
    )
    row: Final = await rows.only_row()

    assert row["spend"] == pytest.approx(_expected_cost(TURN_USAGE, service_tier))
    assert f"{SESSIONS_URL}/{SESSION_ID}" not in api.urls


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

    await _relay_call(
        biller, first, EVENTS_URL, {"events": [INPUT_MESSAGE]}, None, "3c5e7a9c-1d3f-4a6b-9e2c-4b6d8f0a2c38"
    )
    await _relay_call(
        biller, second, EVENTS_URL, {"events": [INPUT_MESSAGE]}, None, "5e7a9c1e-3f5b-4c8d-ae4c-6d8f0b2c4e59"
    )
    billed: Final = sorted([(await first.only_row())["total_tokens"], (await second.only_row())["total_tokens"]])

    assert billed == sorted([TURN_USAGE["total_tokens"], SECOND_TURN_USAGE["total_tokens"]])


@pytest.mark.asyncio
async def test_two_workers_on_one_session_bill_two_different_turns(local_model_cost_map: None):
    api: Final = _AgentsApi(
        turn_pages=(
            (
                _turn("turn_2", CALL_STARTED_AT, "completed", SECOND_TURN_USAGE),
                _turn("turn_1", CALL_STARTED_AT, "completed", TURN_USAGE),
            ),
        ),
        session=_session(),
    )
    redis: Final = _SharedRedis()
    clock: Final = _Clock()
    first: Final = _SpendRows()
    second: Final = _SpendRows()

    await _relay_call(
        OpenAIAgentsTurnBiller(get=api.get, sleep=clock.sleep, clock=clock, claims=TurnClaims(lambda: redis)),
        first,
        EVENTS_URL,
        {"events": [INPUT_MESSAGE]},
        None,
        "3c5e7a9c-1d3f-4a6b-9e2c-4b6d8f0a2c39",
    )
    await _relay_call(
        OpenAIAgentsTurnBiller(get=api.get, sleep=clock.sleep, clock=clock, claims=TurnClaims(lambda: redis)),
        second,
        EVENTS_URL,
        {"events": [INPUT_MESSAGE]},
        None,
        "5e7a9c1e-3f5b-4c8d-ae4c-6d8f0b2c4e5a",
    )
    billed: Final = sorted([(await first.only_row())["total_tokens"], (await second.only_row())["total_tokens"]])

    assert billed == sorted([TURN_USAGE["total_tokens"], SECOND_TURN_USAGE["total_tokens"]])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "redis", [pytest.param(_DownRedis(), id="raises"), pytest.param(_SwallowingRedis(), id="swallows_errors")]
)
async def test_a_shared_claim_store_outage_still_bills_the_turn(
    local_model_cost_map: None, redis: _DownRedis | _SwallowingRedis
):
    clock: Final = _Clock()
    rows: Final = _SpendRows()
    api: Final = _AgentsApi(turn_pages=((_turn("turn_1", CALL_STARTED_AT, "completed", TURN_USAGE),),))

    await _relay_call(
        OpenAIAgentsTurnBiller(get=api.get, sleep=clock.sleep, clock=clock, claims=TurnClaims(lambda: redis)),
        rows,
        SESSIONS_URL,
        {"agent": {"model": AGENT_MODEL}, "input": USER_TEXT},
        _session(),
        "1f3b5d7f-9a2c-4e6b-8d0f-2b4d6f8a0c13",
    )

    assert (await rows.only_row())["total_tokens"] == TURN_USAGE["total_tokens"]


@pytest.mark.asyncio
async def test_concurrent_calls_on_one_worker_claim_a_turn_once_while_the_shared_claim_store_is_down():
    claims: Final = TurnClaims(lambda: _SlowDownRedis())

    granted: Final = await asyncio.gather(claims.claim("turn_1", "call_a"), claims.claim("turn_1", "call_b"))

    assert sorted(granted) == [False, True]


@pytest.mark.asyncio
async def test_an_agent_named_in_the_events_body_never_sets_the_price(local_model_cost_map: None):
    api: Final = _AgentsApi(
        turn_pages=((_turn("turn_1", CALL_STARTED_AT, "completed", TURN_USAGE),),), session=_session()
    )
    rows: Final = _SpendRows()

    await _relay_call(
        _biller(api, _Clock()),
        rows,
        EVENTS_URL,
        {"events": [INPUT_MESSAGE], "agent": {"model": "gpt-4o-mini", "service_tier": "flex"}},
        None,
        "2a4c6e8a-0c2e-4f4a-9b6d-3d5f7b9d1f24",
    )
    row: Final = await rows.only_row()

    assert f"{SESSIONS_URL}/{SESSION_ID}" in api.urls
    assert row["model"] == AGENT_MODEL
    assert row["spend"] == pytest.approx(_expected_cost(TURN_USAGE))


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
        turn_texts={"turn_old": ("an earlier message",)},
    )
    rows: Final = _SpendRows()

    await _relay_call(
        _biller(api, _Clock()),
        rows,
        EVENTS_URL,
        {"events": [INPUT_MESSAGE]},
        None,
        "7a9c1e3a-5b7d-4e0f-8a6e-8f0b2d4e6a70",
    )

    assert (await rows.only_row())["total_tokens"] == TURN_USAGE["total_tokens"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("turn_pages", "status_code", "unreachable"),
    [
        pytest.param(((_turn("turn_1", CALL_STARTED_AT + 1, "in_progress", None),),), 200, False, id="turn_never_ends"),
        pytest.param(
            ((_turn("turn_1", CALL_STARTED_AT + 1, "completed", None),),), 200, False, id="usage_never_arrives"
        ),
        pytest.param(
            ((_turn("turn_old", CALL_STARTED_AT - OPENAI_AGENTS_TURN_CLOCK_SKEW_SECONDS - 1, "completed", TURN_USAGE),),),
            200,
            False,
            id="no_turn_to_claim",
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
        litellm_logging_obj=_logging_obj(
            rows,
            "6a8c0e2a-4c6e-4f8a-9b2c-5a7c9e1a3c46",
            stream=True,
            api_base=SESSIONS_URL,
            litellm_params=_litellm_params(),
        ),
        passthrough_success_handler_obj=PassThroughEndpointLogging(agents_turn_biller=_biller(api, _Clock())),
        url_route=SESSIONS_URL,
        request_body={"agent": {"model": AGENT_MODEL}, "input": USER_TEXT, "stream": True},
        endpoint_type=EndpointType.OPENAI,
        start_time=CALL_TIME,
        raw_bytes=(
            b': keep-alive\n\nevent: agent.session.queued\ndata: {"type": "agent.session.queued"}\n\n',
            f"event: agent.session.created\ndata: {json.dumps(created_event)}\n\n".encode(),
            b'event: agent.turn.completed\ndata: {"type": "agent.turn.completed", "usage": null}\n\n',
        ),
        end_time=CALL_TIME,
    )
    row: Final = await rows.only_row()

    assert f"{SESSIONS_URL}/{SESSION_ID}" not in api.urls
    assert row["model"] == AGENT_MODEL
    assert row["total_tokens"] == TURN_USAGE["total_tokens"]
    assert row["spend"] == pytest.approx(_expected_cost(TURN_USAGE))


class _GatedClock(_Clock):
    def __init__(self) -> None:
        super().__init__()
        self.gate: Final = asyncio.Event()

    async def sleep(self, seconds: float) -> None:
        await self.gate.wait()
        await super().sleep(seconds)


def _as_new_request(coroutine: Coroutine[object, object, object]) -> "asyncio.Future[object]":
    return contextvars.Context().run(asyncio.ensure_future, coroutine)


@pytest.mark.asyncio
async def test_a_key_at_its_parallel_limit_can_call_again_while_its_agent_turn_bills(local_model_cost_map: None):
    limiter: Final = PROXY_MaxParallelRequestsHandler_v3(internal_usage_cache=InternalUsageCache(DualCache()))
    key: Final = UserAPIKeyAuth(api_key=hash_token("sk-agents-parallel"), max_parallel_requests=1)
    api: Final = _AgentsApi(
        turn_pages=((_turn("turn_1", CALL_STARTED_AT + 1, "completed", TURN_USAGE),),), session=_session()
    )
    clock: Final = _GatedClock()
    rows: Final = _SpendRows()
    released: Final = asyncio.Event()

    async def release_request_slot(logging_obj: LiteLLMLoggingObj) -> None:
        await limiter.async_release_request_parallel_slot()
        released.set()

    biller: Final = OpenAIAgentsTurnBiller(
        get=api.get, sleep=clock.sleep, clock=clock, release_request_slot=release_request_slot
    )

    async def agent_turn(litellm_call_id: str) -> None:
        await limiter.async_pre_call_hook(
            user_api_key_dict=key,
            cache=DualCache(),
            data={"litellm_call_id": litellm_call_id},
            call_type="pass_through_endpoint",
        )
        await _relay_call(biller, rows, EVENTS_URL, {"events": [INPUT_MESSAGE]}, None, litellm_call_id)

    await _as_new_request(agent_turn("0c2e4a6c-8e0a-4c2e-9a4c-6e8a0c2e4a61"))
    await asyncio.wait_for(released.wait(), timeout=5)
    await _as_new_request(
        limiter.async_pre_call_hook(
            user_api_key_dict=key,
            cache=DualCache(),
            data={"litellm_call_id": "1d3f5b7d-9f1b-4d3f-8b5d-7f9b1d3f5b72"},
            call_type="pass_through_endpoint",
        )
    )

    assert api.turn_polls == 0
    clock.gate.set()
    row: Final = await rows.only_row()
    assert row["spend"] == pytest.approx(_expected_cost(TURN_USAGE))


@pytest.mark.asyncio
async def test_the_polls_send_the_passthrough_key_and_the_callers_openai_scope(
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
        session: Final = upstream.get(f"{SESSIONS_URL}/{SESSION_ID}").respond(json=_session())
        subagents: Final = upstream.get(SUBAGENTS_URL).respond(json=_page())
        items: Final = upstream.get(ROOT_ITEMS_URL).respond(json=_page())
        messages: Final = upstream.get(f"{SESSIONS_URL}/{SESSION_ID}/turns/turn_1/items?order=asc&limit=100").respond(
            json=_user_messages((USER_TEXT,))
        )
        await _relay_call(
            OpenAIAgentsTurnBiller(get=live_agents_api_get, sleep=clock.sleep, clock=clock),
            rows,
            EVENTS_URL,
            {"events": [INPUT_MESSAGE]},
            None,
            "8c0e2a4c-6e8a-4b0c-9d4e-7c9e1a3c5e68",
            request_headers={
                "x-pass-OpenAI-Project": "proj_agents_scoped",
                "x-pass-openai-organization": "org_agents_scoped",
                "x-pass-authorization": "Bearer sk-caller-override",
                "openai-project": "proj_not_forwarded",
            },
        )
        row: Final = await rows.only_row()
    litellm.in_memory_llm_clients_cache.flush_cache()

    for poll in (route.calls.last.request for route in (turns, session, subagents, items, messages)):
        assert poll.headers["authorization"] == "Bearer sk-agents-turns-poll"
        assert poll.headers["openai-beta"] == "agents=v1"
        assert poll.headers["openai-project"] == "proj_agents_scoped"
        assert poll.headers["openai-organization"] == "org_agents_scoped"
    assert row["spend"] == pytest.approx(_expected_cost(TURN_USAGE))


async def test_a_poll_redirected_off_openai_never_follows_it_with_the_proxys_key(
    local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-agents-turns-poll")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    litellm.in_memory_llm_clients_cache.flush_cache()
    clock: Final = _Clock()
    rows: Final = _SpendRows()
    elsewhere_url: Final = "https://agents-elsewhere.example/v1/turns"
    turns_page: Final = {"object": "list", "data": [_turn("turn_1", CALL_STARTED_AT, "completed", TURN_USAGE)]}

    with respx.mock(assert_all_called=False) as upstream:
        upstream.get(TURNS_URL).respond(307, headers={"location": elsewhere_url})
        elsewhere: Final = upstream.get(elsewhere_url).respond(json=turns_page)
        upstream.get(f"{SESSIONS_URL}/{SESSION_ID}").respond(json=_session())
        upstream.get(SUBAGENTS_URL).respond(json=_page())
        upstream.get(ROOT_ITEMS_URL).respond(json=_page())
        upstream.get(f"{SESSIONS_URL}/{SESSION_ID}/turns/turn_1/items?order=asc&limit=100").respond(
            json=_user_messages((USER_TEXT,))
        )
        await _relay_call(
            OpenAIAgentsTurnBiller(get=live_agents_api_get, sleep=clock.sleep, clock=clock),
            rows,
            EVENTS_URL,
            {"events": [INPUT_MESSAGE]},
            None,
            "5b7d1e2f-3a4c-4d6e-8f9a-0b1c2d3e4f5a",
        )
        row: Final = await rows.only_row()
    litellm.in_memory_llm_clients_cache.flush_cache()

    assert not elsewhere.called
    assert (row["spend"] or 0.0) == 0.0


async def _cancelled_sleep(seconds: float) -> None:
    raise asyncio.CancelledError


async def _failing_sleep(seconds: float) -> None:
    raise RuntimeError("event loop is closing")


@pytest.mark.asyncio
async def test_a_billing_task_that_is_cancelled_reports_the_lost_row(
    local_model_cost_map: None, caplog: pytest.LogCaptureFixture
):
    api: Final = _AgentsApi(turn_pages=((_turn("turn_1", CALL_STARTED_AT, "completed", TURN_USAGE),),))
    rows: Final = _SpendRows()

    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        await _relay_call(
            OpenAIAgentsTurnBiller(get=api.get, sleep=_cancelled_sleep, clock=_Clock()),
            rows,
            EVENTS_URL,
            {"events": [INPUT_MESSAGE]},
            None,
            "0e2a4c6e-8a0c-4e2d-8b6c-9e1a3c5e7a80",
        )
        for _ in range(50):
            await asyncio.sleep(0)

    assert rows.rows == []
    assert any("turn billing cancelled, the call logged no row" in message for message in caplog.messages)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("sleep_fails", "replies"),
    [
        pytest.param(True, {}, id="sleep_fails"),
        pytest.param(False, {TURNS_URL: (ValueError("unexpected turns reply"),)}, id="poll_raises"),
        pytest.param(
            False, {f"{SESSIONS_URL}/{SESSION_ID}": (ValueError("unexpected session reply"),)}, id="costing_raises"
        ),
    ],
)
async def test_a_poll_that_breaks_unexpectedly_still_logs_the_call_at_zero(
    local_model_cost_map: None, sleep_fails: bool, replies: Mapping[str, Sequence[_Reply]]
):
    api: Final = _AgentsApi(
        turn_pages=((_turn("turn_1", CALL_STARTED_AT, "completed", TURN_USAGE),),), replies=replies
    )
    clock: Final = _Clock()
    rows: Final = _SpendRows()

    await _relay_call(
        OpenAIAgentsTurnBiller(get=api.get, sleep=_failing_sleep if sleep_fails else clock.sleep, clock=clock),
        rows,
        EVENTS_URL,
        {"events": [INPUT_MESSAGE]},
        None,
        "0e2a4c6e-8a0c-4e2d-8b6c-9e1a3c5e7a81",
    )
    row: Final = await rows.only_row()

    assert not row["spend"]
    assert row["total_tokens"] == 0


@pytest.mark.asyncio
async def test_a_billed_row_keeps_the_upstream_api_base(local_model_cost_map: None):
    api: Final = _AgentsApi(
        turn_pages=((_turn("turn_1", CALL_STARTED_AT, "completed", TURN_USAGE),),), session=_session()
    )
    rows: Final = _SpendRows()

    await _relay_call(
        _biller(api, _Clock()),
        rows,
        EVENTS_URL,
        {"events": [INPUT_MESSAGE]},
        None,
        "1a3c5e7a-9c1e-4a3c-8e5a-7c9e1a3c5e71",
    )
    row: Final = await rows.only_row()
    standard_logging_object: Final = rows.standard_logging_objects[0]

    assert row["spend"] == pytest.approx(_expected_cost(TURN_USAGE))
    assert row["api_base"] == EVENTS_URL
    assert isinstance(standard_logging_object, dict)
    assert standard_logging_object["api_base"] == EVENTS_URL


async def _blocked_sleep(seconds: float) -> None:
    await asyncio.Event().wait()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("turn", "billed_usage"),
    [
        pytest.param(_turn("turn_1", CALL_STARTED_AT + 1, "completed", TURN_USAGE), TURN_USAGE, id="settled"),
        pytest.param(
            _turn("turn_1", CALL_STARTED_AT + 1, "in_progress", SECOND_TURN_USAGE), SECOND_TURN_USAGE, id="running"
        ),
        pytest.param(_turn("turn_1", CALL_STARTED_AT + 1, "in_progress", None), None, id="no_usage_yet"),
    ],
)
async def test_a_proxy_shutdown_logs_a_waiting_row_with_the_usage_reported_so_far(
    local_model_cost_map: None,
    caplog: pytest.LogCaptureFixture,
    turn: Mapping[str, object],
    billed_usage: Mapping[str, object] | None,
):
    api: Final = _AgentsApi(turn_pages=((turn,),), session=_session())
    rows: Final = _SpendRows()
    biller: Final = OpenAIAgentsTurnBiller(get=api.get, sleep=_blocked_sleep, clock=_Clock())

    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        await _relay_call(
            biller, rows, EVENTS_URL, {"events": [INPUT_MESSAGE]}, None, "2b4d6f8b-0d2f-4b4d-9f6b-8d0f2b4d6f82"
        )
        await biller.drain(timeout=5)
    row: Final = await rows.only_row()

    assert row["total_tokens"] == (0 if billed_usage is None else billed_usage["total_tokens"])
    assert (row["spend"] or 0.0) == pytest.approx(0.0 if billed_usage is None else _expected_cost(billed_usage))
    assert (billed_usage == TURN_USAGE) != any(SESSION_ID in message for message in caplog.messages)


async def _unanswered_get(base_url: str, path: str, scope_headers: Mapping[str, str]) -> httpx.Response:
    _ = await asyncio.Event().wait()
    raise AssertionError(f"{base_url}/{path} never answers")


@pytest.mark.asyncio
async def test_a_shutdown_that_runs_out_of_time_logs_a_waiting_row_at_zero_before_it_ends(
    local_model_cost_map: None, caplog: pytest.LogCaptureFixture
):
    rows: Final = _SpendRows()
    biller: Final = OpenAIAgentsTurnBiller(get=_unanswered_get, sleep=_blocked_sleep, clock=_Clock())

    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        await _relay_call(
            biller, rows, EVENTS_URL, {"events": [INPUT_MESSAGE]}, None, "3c5e7a9c-1e3a-4c5e-8a7c-9e1a3c5e7a93"
        )
        await biller.drain(timeout=0.2)
        logged_before_drain_ended: Final = rows.logged.is_set()
    row: Final = await rows.only_row()

    assert logged_before_drain_ended
    assert row["total_tokens"] == 0
    assert not row["spend"]
    assert any(SESSION_ID in message and "shut down" in message for message in caplog.messages)


@pytest.mark.asyncio
async def test_transient_poll_failures_still_bill_the_turn(local_model_cost_map: None):
    api: Final = _AgentsApi(
        turn_pages=((),),
        session=_session(),
        replies={
            TURNS_URL: (
                503,
                500,
                httpx.ConnectError("connection refused"),
                openai.OpenAIError("The api_key client option must be set"),
                LiteLLMOpenAIError(status_code=401, message="credential refresh failed"),
                429,
                408,
                _page(_turn("turn_1", CALL_STARTED_AT + 1, "completed", TURN_USAGE)),
            )
        },
    )
    rows: Final = _SpendRows()

    await _relay_call(
        _biller(api, _Clock()),
        rows,
        EVENTS_URL,
        {"events": [INPUT_MESSAGE]},
        None,
        "3c5e7a9c-1e3a-4c5e-8a7c-9e1a3c5e7a93",
    )
    row: Final = await rows.only_row()

    assert row["spend"] == pytest.approx(_expected_cost(TURN_USAGE))


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 403, 404])
async def test_a_poll_the_upstream_refuses_logs_the_call_at_zero_without_retrying(
    local_model_cost_map: None, status_code: int
):
    api: Final = _AgentsApi(turn_pages=((),), session=_session(), replies={TURNS_URL: (status_code,)})
    rows: Final = _SpendRows()

    await _relay_call(
        _biller(api, _Clock()),
        rows,
        EVENTS_URL,
        {"events": [INPUT_MESSAGE]},
        None,
        "4d6f8b0d-2f4b-4d6f-9b8d-0f2b4d6f8ba4",
    )
    row: Final = await rows.only_row()

    assert api.urls.count(TURNS_URL) == 1
    assert not row["spend"]


@pytest.mark.asyncio
async def test_usage_that_arrives_minutes_after_the_turn_completes_is_billed(local_model_cost_map: None):
    completed_without_usage: Final = (_turn("turn_1", CALL_STARTED_AT + 1, "completed", None),)
    api: Final = _AgentsApi(
        turn_pages=(*(completed_without_usage,) * 10, (_turn("turn_1", CALL_STARTED_AT + 1, "completed", TURN_USAGE),)),
        session=_session(),
    )
    clock: Final = _Clock()
    rows: Final = _SpendRows()

    await _relay_call(
        _biller(api, clock), rows, EVENTS_URL, {"events": [INPUT_MESSAGE]}, None, "5e7a9c1e-3a5c-4e7a-8c9e-1a3c5e7a9cb5"
    )
    row: Final = await rows.only_row()

    assert clock.now > 300
    assert row["spend"] == pytest.approx(_expected_cost(TURN_USAGE))


@pytest.mark.asyncio
async def test_a_turn_still_running_at_the_deadline_bills_the_usage_it_reported(
    local_model_cost_map: None, caplog: pytest.LogCaptureFixture
):
    api: Final = _AgentsApi(
        turn_pages=((_turn("turn_1", CALL_STARTED_AT + 1, "in_progress", SECOND_TURN_USAGE),),), session=_session()
    )
    rows: Final = _SpendRows()

    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        await _relay_call(
            _biller(api, _Clock()),
            rows,
            EVENTS_URL,
            {"events": [INPUT_MESSAGE]},
            None,
            "6f8b0d2f-4b6d-4f8b-9d0f-2b4d6f8b0dc6",
        )
        row: Final = await rows.only_row()

    assert row["spend"] == pytest.approx(_expected_cost(SECOND_TURN_USAGE))
    assert any(SESSION_ID in message and "did not settle" in message for message in caplog.messages)


def _subagent_api(subagent_turns: Sequence[Mapping[str, object]]) -> _AgentsApi:
    return _AgentsApi(
        turn_pages=((_turn("turn_1", CALL_STARTED_AT + 1, "completed", TURN_USAGE),),),
        session=_session(),
        replies={
            SUBAGENTS_URL: (_page({"id": "subagent_1", "object": "agent.session.subagent"}),),
            SUBAGENT_TURNS_URL: tuple(_page(turn) for turn in subagent_turns),
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "subagent_turns",
    [
        pytest.param((_turn("subturn_1", CALL_STARTED_AT + 3, "completed", SECOND_TURN_USAGE),), id="settled"),
        pytest.param(
            (
                _turn("subturn_1", CALL_STARTED_AT + 3, "in_progress", None),
                _turn("subturn_1", CALL_STARTED_AT + 3, "completed", SECOND_TURN_USAGE),
            ),
            id="still_running",
        ),
    ],
)
async def test_subagent_turns_the_call_started_are_billed_at_the_session_agent_model(
    local_model_cost_map: None, subagent_turns: Sequence[Mapping[str, object]]
):
    rows: Final = _SpendRows()

    await _relay_call(
        _biller(_subagent_api(subagent_turns), _Clock()),
        rows,
        EVENTS_URL,
        {"events": [INPUT_MESSAGE]},
        None,
        "7a9c1e3a-5c7e-4a9c-8e1a-3c5e7a9c1ed7",
    )
    row: Final = await rows.only_row()

    assert row["model"] == AGENT_MODEL
    assert row["prompt_tokens"] == TURN_USAGE["input_tokens"] + SECOND_TURN_USAGE["input_tokens"]
    assert row["total_tokens"] == TURN_USAGE["total_tokens"] + SECOND_TURN_USAGE["total_tokens"]
    assert row["spend"] == pytest.approx(_expected_cost(TURN_USAGE) + _expected_cost(SECOND_TURN_USAGE))


@pytest.mark.asyncio
async def test_a_subagent_turn_from_before_the_call_is_never_billed_to_it(local_model_cost_map: None):
    rows: Final = _SpendRows()

    await _relay_call(
        _biller(_subagent_api((_turn("subturn_0", CALL_STARTED_AT - 5, "completed", SECOND_TURN_USAGE),)), _Clock()),
        rows,
        EVENTS_URL,
        {"events": [INPUT_MESSAGE]},
        None,
        "8b0d2f4b-6d8f-4b0d-9f2b-4d6f8b0d2fe8",
    )
    row: Final = await rows.only_row()

    assert row["total_tokens"] == TURN_USAGE["total_tokens"]
    assert row["spend"] == pytest.approx(_expected_cost(TURN_USAGE))


@pytest.mark.asyncio
async def test_a_subagent_turn_a_later_root_turn_started_is_never_billed_to_the_earlier_call(local_model_cost_map: None):
    api: Final = _AgentsApi(
        turn_pages=(
            (
                _turn("turn_2", CALL_STARTED_AT + 10, "completed", TURN_USAGE),
                _turn("turn_1", CALL_STARTED_AT + 1, "completed", TURN_USAGE),
            ),
        ),
        session=_session(),
        replies={
            SUBAGENTS_URL: (_page({"id": "subagent_1", "object": "agent.session.subagent"}),),
            SUBAGENT_TURNS_URL: (
                _page(
                    _turn("subturn_2", CALL_STARTED_AT + 12, "completed", TURN_USAGE),
                    _turn("subturn_1", CALL_STARTED_AT + 3, "completed", SECOND_TURN_USAGE),
                ),
            ),
        },
    )
    rows: Final = _SpendRows()

    await _relay_call(
        _biller(api, _Clock()),
        rows,
        EVENTS_URL,
        {"events": [INPUT_MESSAGE]},
        None,
        "0d2f4b6d-8f0b-4d2f-9b4d-6f8b0d2f4b6d",
    )
    row: Final = await rows.only_row()

    assert row["total_tokens"] == TURN_USAGE["total_tokens"] + SECOND_TURN_USAGE["total_tokens"]
    assert row["spend"] == pytest.approx(_expected_cost(TURN_USAGE) + _expected_cost(SECOND_TURN_USAGE))


@pytest.mark.asyncio
async def test_web_searches_in_a_billed_turn_add_their_per_call_price(local_model_cost_map: None):
    web_search: Final = {"type": "web_search_call", "id": "ws_1", "status": "completed"}
    api: Final = _AgentsApi(
        turn_pages=((_turn("turn_1", CALL_STARTED_AT + 1, "completed", TURN_USAGE),),),
        session=_session(),
        replies={
            ROOT_ITEMS_URL: (
                {"object": "list", "data": [web_search, {"type": "message"}], "has_more": True, "last_id": "item_2"},
            ),
            f"{ROOT_ITEMS_URL}&after=item_2": (_page(web_search),),
        },
    )
    rows: Final = _SpendRows()
    per_search: Final = litellm.model_cost[AGENT_MODEL]["search_context_cost_per_query"]["search_context_size_medium"]

    await _relay_call(
        _biller(api, _Clock()),
        rows,
        EVENTS_URL,
        {"events": [INPUT_MESSAGE]},
        None,
        "9c1e3a5c-7e9a-4c1e-8a3c-5e7a9c1e3af9",
    )
    row: Final = await rows.only_row()

    assert per_search > 0
    assert row["spend"] == pytest.approx(_expected_cost(TURN_USAGE) + 2 * per_search)


@pytest.mark.asyncio
async def test_transient_failures_reading_a_settled_turns_agent_and_searches_still_bill_them(
    local_model_cost_map: None,
):
    web_search: Final = {"type": "web_search_call", "id": "ws_1", "status": "completed"}
    api: Final = _AgentsApi(
        turn_pages=((_turn("turn_1", CALL_STARTED_AT + 1, "completed", TURN_USAGE),),),
        replies={
            f"{SESSIONS_URL}/{SESSION_ID}": (503, httpx.ConnectError("connection refused"), _session()),
            ROOT_ITEMS_URL: (502, _page(web_search)),
        },
    )
    rows: Final = _SpendRows()
    per_search: Final = litellm.model_cost[AGENT_MODEL]["search_context_cost_per_query"]["search_context_size_medium"]

    await _relay_call(
        _biller(api, _Clock()),
        rows,
        EVENTS_URL,
        {"events": [INPUT_MESSAGE]},
        None,
        "1e3a5c7e-9a1c-4e3a-8c5e-7a9c1e3a5c7e",
    )
    row: Final = await rows.only_row()

    assert row["model"] == AGENT_MODEL
    assert row["spend"] == pytest.approx(_expected_cost(TURN_USAGE) + per_search)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "reads"),
    [pytest.param(404, 1, id="refused"), pytest.param(503, OPENAI_AGENTS_SETTLED_FETCH_ATTEMPTS, id="still_failing")],
)
async def test_a_settled_turns_agent_read_stops_retrying_when_refused_or_out_of_attempts(
    local_model_cost_map: None, status_code: int, reads: int
):
    session_url: Final = f"{SESSIONS_URL}/{SESSION_ID}"
    api: Final = _AgentsApi(
        turn_pages=((_turn("turn_1", CALL_STARTED_AT + 1, "completed", TURN_USAGE),),),
        replies={session_url: (*([status_code] * reads), _session())},
    )
    rows: Final = _SpendRows()

    await _relay_call(
        _biller(api, _Clock()),
        rows,
        EVENTS_URL,
        {"events": [INPUT_MESSAGE]},
        None,
        "2f4b6d8f-0b2d-4f4b-8d6f-8b0d2f4b6d8f",
    )
    row: Final = await rows.only_row()

    assert api.urls.count(session_url) == reads
    assert row["spend"] == 0


@pytest.mark.asyncio
async def test_a_turn_stamped_before_the_call_by_a_skewed_clock_is_still_billed(local_model_cost_map: None):
    api: Final = _AgentsApi(
        turn_pages=((_turn("turn_1", CALL_STARTED_AT - OPENAI_AGENTS_TURN_CLOCK_SKEW_SECONDS, "completed", TURN_USAGE),),),
        session=_session(),
    )
    rows: Final = _SpendRows()

    await _relay_call(
        _biller(api, _Clock()),
        rows,
        EVENTS_URL,
        {"events": [INPUT_MESSAGE]},
        None,
        "1a3c5e7a-9c1e-4a3c-8e5a-7c9e1a3c5e01",
    )

    assert (await rows.only_row())["spend"] == pytest.approx(_expected_cost(TURN_USAGE))


@pytest.mark.asyncio
async def test_a_message_merged_into_another_calls_turn_never_takes_a_later_calls_turn(local_model_cost_map: None):
    api: Final = _AgentsApi(
        turn_pages=(
            (
                _turn("turn_2", CALL_STARTED_AT + 5, "completed", SECOND_TURN_USAGE),
                _turn("turn_1", CALL_STARTED_AT, "completed", TURN_USAGE),
            ),
        ),
        session=_session(),
        turn_texts={"turn_1": ("first message", "second message"), "turn_2": ("third message",)},
    )
    biller: Final = _biller(api, _Clock())
    rows: Final = tuple(_SpendRows() for _ in range(3))

    for call_rows, text, call_id in zip(
        rows,
        ("first message", "second message", "third message"),
        (
            "2b4d6f8b-0d2f-4b4d-8f6b-8d0f2b4d6f01",
            "2b4d6f8b-0d2f-4b4d-8f6b-8d0f2b4d6f02",
            "2b4d6f8b-0d2f-4b4d-8f6b-8d0f2b4d6f03",
        ),
        strict=True,
    ):
        await _relay_call(biller, call_rows, EVENTS_URL, {"events": [_input_message(text)]}, None, call_id)
    first, merged, third = [(await call_rows.only_row())["total_tokens"] for call_rows in rows]

    assert first == TURN_USAGE["total_tokens"]
    assert merged == 0
    assert third == SECOND_TURN_USAGE["total_tokens"]


@pytest.mark.asyncio
async def test_a_turn_whose_messages_cannot_be_read_is_never_claimed_by_the_call(local_model_cost_map: None):
    api: Final = _AgentsApi(
        turn_pages=(
            (
                _turn("turn_1", CALL_STARTED_AT + 1, "completed", TURN_USAGE),
                _turn("turn_other", CALL_STARTED_AT, "completed", SECOND_TURN_USAGE),
            ),
        ),
        session=_session(),
        replies={f"{SESSIONS_URL}/{SESSION_ID}/turns/turn_other/items?order=asc&limit=100": (404,)},
    )
    rows: Final = _SpendRows()

    await _relay_call(
        _biller(api, _Clock()),
        rows,
        EVENTS_URL,
        {"events": [INPUT_MESSAGE]},
        None,
        "3c5e7a9c-1e3a-4c5e-9a7c-9e1a3c5e7a01",
    )
    row: Final = await rows.only_row()

    assert row["total_tokens"] == TURN_USAGE["total_tokens"]
    assert row["spend"] == pytest.approx(_expected_cost(TURN_USAGE))


@pytest.mark.asyncio
async def test_a_failed_read_of_the_calls_own_turn_never_bills_a_later_turn_with_the_same_text(
    local_model_cost_map: None,
):
    api: Final = _AgentsApi(
        turn_pages=(
            (
                _turn("turn_2", CALL_STARTED_AT + 1, "completed", SECOND_TURN_USAGE),
                _turn("turn_1", CALL_STARTED_AT, "completed", TURN_USAGE),
            ),
        ),
        session=_session(),
        replies={
            f"{SESSIONS_URL}/{SESSION_ID}/turns/turn_1/items?order=asc&limit=100": (503, _user_messages((USER_TEXT,)))
        },
    )
    rows: Final = _SpendRows()

    await _relay_call(
        _biller(api, _Clock()),
        rows,
        EVENTS_URL,
        {"events": [INPUT_MESSAGE]},
        None,
        "3c5e7a9c-1e3a-4c5e-9a7c-9e1a3c5e7a02",
    )
    row: Final = await rows.only_row()

    assert row["total_tokens"] == TURN_USAGE["total_tokens"]
    assert row["spend"] == pytest.approx(_expected_cost(TURN_USAGE))


def _one_worker(redis: _SharedRedis) -> tuple[TurnClaims, TurnClaims]:
    claims: Final = TurnClaims(lambda: None)
    return claims, claims


def _two_workers(redis: _SharedRedis) -> tuple[TurnClaims, TurnClaims]:
    return TurnClaims(lambda: redis), TurnClaims(lambda: redis)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "workers", [pytest.param(_one_worker, id="one_worker"), pytest.param(_two_workers, id="two_workers")]
)
async def test_an_earlier_calls_turn_that_cannot_be_read_again_never_blocks_billing_the_next_call(
    local_model_cost_map: None, workers: Callable[[_SharedRedis], tuple[TurnClaims, TurnClaims]]
):
    api: Final = _AgentsApi(
        turn_pages=(
            (
                _turn("turn_2", CALL_STARTED_AT + 1, "completed", SECOND_TURN_USAGE),
                _turn("turn_1", CALL_STARTED_AT, "completed", TURN_USAGE),
            ),
        ),
        session=_session(),
        turn_texts={"turn_2": ("second message",)},
        replies={
            f"{SESSIONS_URL}/{SESSION_ID}/turns/turn_1/items?order=asc&limit=100": (
                _user_messages(("first message",)),
                503,
            )
        },
    )
    clock: Final = _Clock()
    first_claims, second_claims = workers(_SharedRedis())
    first: Final = _SpendRows()
    second: Final = _SpendRows()

    await _relay_call(
        OpenAIAgentsTurnBiller(get=api.get, sleep=clock.sleep, clock=clock, claims=first_claims),
        first,
        EVENTS_URL,
        {"events": [_input_message("first message")]},
        None,
        "3c5e7a9c-1e3a-4c5e-9a7c-9e1a3c5e7a04",
    )
    await _relay_call(
        OpenAIAgentsTurnBiller(get=api.get, sleep=clock.sleep, clock=clock, claims=second_claims),
        second,
        EVENTS_URL,
        {"events": [_input_message("second message")]},
        None,
        "3c5e7a9c-1e3a-4c5e-9a7c-9e1a3c5e7a05",
    )

    assert (await first.only_row())["total_tokens"] == TURN_USAGE["total_tokens"]
    assert (await second.only_row())["total_tokens"] == SECOND_TURN_USAGE["total_tokens"]


@pytest.mark.asyncio
async def test_an_unreadable_turn_items_page_never_logs_the_conversation_text(
    local_model_cost_map: None, caplog: pytest.LogCaptureFixture
):
    conversation_text: Final = "my pin is 4821"
    api: Final = _AgentsApi(
        turn_pages=((_turn("turn_1", CALL_STARTED_AT, "completed", TURN_USAGE),),),
        session=_session(),
        replies={
            f"{SESSIONS_URL}/{SESSION_ID}/turns/turn_1/items?order=asc&limit=100": (
                _page({"type": "message", "role": "user", "content": {"text": conversation_text}}),
            )
        },
    )
    rows: Final = _SpendRows()

    with caplog.at_level(logging.WARNING):
        await _relay_call(
            _biller(api, _Clock()),
            rows,
            EVENTS_URL,
            {"events": [INPUT_MESSAGE]},
            None,
            "4d6f8b0d-2f4b-4d6f-8b0d-0f2b4d6f8b01",
        )
        _ = await rows.only_row()

    assert "unreadable body" in caplog.text
    assert conversation_text not in caplog.text


@pytest.mark.asyncio
async def test_two_workers_without_a_shared_claim_store_bill_each_message_its_own_turn(local_model_cost_map: None):
    api: Final = _AgentsApi(
        turn_pages=(
            (
                _turn("turn_2", CALL_STARTED_AT, "completed", SECOND_TURN_USAGE),
                _turn("turn_1", CALL_STARTED_AT, "completed", TURN_USAGE),
            ),
        ),
        session=_session(),
        turn_texts={"turn_1": ("message on worker one",), "turn_2": ("message on worker two",)},
    )
    clock: Final = _Clock()
    first: Final = _SpendRows()
    second: Final = _SpendRows()

    await _relay_call(
        _biller(api, clock),
        first,
        EVENTS_URL,
        {"events": [_input_message("message on worker one")]},
        None,
        "3c5e7a9c-1e3a-4c5e-8a7c-9e1a3c5e7a01",
    )
    await _relay_call(
        _biller(api, clock),
        second,
        EVENTS_URL,
        {"events": [_input_message("message on worker two")]},
        None,
        "3c5e7a9c-1e3a-4c5e-8a7c-9e1a3c5e7a02",
    )

    assert (await first.only_row())["total_tokens"] == TURN_USAGE["total_tokens"]
    assert (await second.only_row())["total_tokens"] == SECOND_TURN_USAGE["total_tokens"]


@pytest.mark.asyncio
async def test_a_subagent_closed_before_the_turn_started_is_never_polled(local_model_cost_map: None):
    closed_turns_url: Final = f"{SESSIONS_URL}/{SESSION_ID}/subagents/subagent_0/turns?order=desc&limit=100"
    api: Final = _AgentsApi(
        turn_pages=((_turn("turn_1", CALL_STARTED_AT + 1, "completed", TURN_USAGE),),),
        session=_session(),
        replies={
            SUBAGENTS_URL: (
                _page(
                    {"id": "subagent_1", "object": "agent.session.subagent", "closed_at": None},
                    {"id": "subagent_0", "object": "agent.session.subagent", "closed_at": CALL_STARTED_AT - 30},
                ),
            ),
            SUBAGENT_TURNS_URL: (_page(_turn("subturn_1", CALL_STARTED_AT + 3, "completed", SECOND_TURN_USAGE)),),
            closed_turns_url: (_page(_turn("subturn_0", CALL_STARTED_AT + 3, "completed", SECOND_TURN_USAGE)),),
        },
    )
    rows: Final = _SpendRows()

    await _relay_call(
        _biller(api, _Clock()),
        rows,
        EVENTS_URL,
        {"events": [INPUT_MESSAGE]},
        None,
        "4d6f8b0d-2f4b-4d6f-9b8d-0f2b4d6f8b01",
    )
    row: Final = await rows.only_row()

    assert closed_turns_url not in api.urls
    assert row["total_tokens"] == TURN_USAGE["total_tokens"] + SECOND_TURN_USAGE["total_tokens"]


@pytest.mark.asyncio
async def test_a_turn_row_on_the_legacy_limiter_frees_its_slot_once(local_model_cost_map: None):
    limiter: Final = PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()), clock=lambda: CALL_TIME
    )
    proxy_logging: Final = ProxyLogging(user_api_key_cache=DualCache())
    proxy_logging.proxy_hook_mapping["parallel_request_limiter"] = limiter
    key: Final = UserAPIKeyAuth(api_key=hash_token("sk-agents-legacy-parallel"), max_parallel_requests=1)
    api: Final = _AgentsApi(
        turn_pages=((_turn("turn_1", CALL_STARTED_AT + 1, "completed", TURN_USAGE),),), session=_session()
    )
    clock: Final = _GatedClock()
    rows: Final = _SpendRows()
    released: Final = asyncio.Event()

    async def release_request_slot(logging_obj: LiteLLMLoggingObj) -> None:
        await proxy_logging.arelease_slot_before_logging(logging_obj)
        released.set()

    async def call(litellm_call_id: str) -> None:
        await limiter.async_pre_call_hook(
            user_api_key_dict=key,
            cache=DualCache(),
            data={"litellm_call_id": litellm_call_id},
            call_type="pass_through_endpoint",
        )

    await call("5e7a9c1e-3a5c-4e7a-8c9e-1a3c5e7a9c01")
    await _relay_call(
        OpenAIAgentsTurnBiller(get=api.get, sleep=clock.sleep, clock=clock, release_request_slot=release_request_slot),
        rows,
        EVENTS_URL,
        {"events": [INPUT_MESSAGE]},
        None,
        "5e7a9c1e-3a5c-4e7a-8c9e-1a3c5e7a9c01",
        metadata={"user_api_key": key.api_key, "user_api_key_model_max_budget": key.model_max_budget},
        callbacks=(limiter,),
    )
    await asyncio.wait_for(released.wait(), timeout=5)
    await call("5e7a9c1e-3a5c-4e7a-8c9e-1a3c5e7a9c02")
    clock.gate.set()
    _ = await rows.only_row()

    with pytest.raises(ProxyRateLimitError):
        await call("5e7a9c1e-3a5c-4e7a-8c9e-1a3c5e7a9c03")


class _LimiterClock:
    def __init__(self) -> None:
        self.now = CALL_TIME

    def __call__(self) -> datetime:
        return self.now


@pytest.mark.asyncio
async def test_a_turn_row_the_legacy_limiter_sees_a_minute_later_leaves_the_key_free(local_model_cost_map: None):
    limiter_clock: Final = _LimiterClock()
    limiter: Final = PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()), clock=limiter_clock
    )
    proxy_logging: Final = ProxyLogging(user_api_key_cache=DualCache())
    proxy_logging.proxy_hook_mapping["parallel_request_limiter"] = limiter
    key: Final = UserAPIKeyAuth(
        api_key=hash_token("sk-agents-legacy-next-minute"),
        max_parallel_requests=1,
        user_id="agents-user",
        user_rpm_limit=1,
    )
    api: Final = _AgentsApi(
        turn_pages=((_turn("turn_1", CALL_STARTED_AT + 1, "completed", TURN_USAGE),),), session=_session()
    )
    clock: Final = _GatedClock()
    rows: Final = _SpendRows()
    released: Final = asyncio.Event()

    async def release_request_slot(logging_obj: LiteLLMLoggingObj) -> None:
        await proxy_logging.arelease_slot_before_logging(logging_obj)
        released.set()

    async def call(litellm_call_id: str) -> None:
        await limiter.async_pre_call_hook(
            user_api_key_dict=key,
            cache=DualCache(),
            data={"litellm_call_id": litellm_call_id},
            call_type="pass_through_endpoint",
        )

    await call("7a9c1e3a-5c7e-4a9c-8e1a-3c5e7a9c1e01")
    await _relay_call(
        OpenAIAgentsTurnBiller(get=api.get, sleep=clock.sleep, clock=clock, release_request_slot=release_request_slot),
        rows,
        EVENTS_URL,
        {"events": [INPUT_MESSAGE]},
        None,
        "7a9c1e3a-5c7e-4a9c-8e1a-3c5e7a9c1e01",
        metadata={
            "user_api_key": key.api_key,
            "user_api_key_user_id": key.user_id,
            "user_api_key_model_max_budget": key.model_max_budget,
        },
        callbacks=(limiter,),
    )
    await asyncio.wait_for(released.wait(), timeout=5)
    limiter_clock.now = CALL_TIME + timedelta(minutes=1)
    clock.gate.set()
    _ = await rows.only_row()

    await call("7a9c1e3a-5c7e-4a9c-8e1a-3c5e7a9c1e02")
    with pytest.raises(ProxyRateLimitError):
        await call("7a9c1e3a-5c7e-4a9c-8e1a-3c5e7a9c1e03")


async def _failing_release(logging_obj: LiteLLMLoggingObj) -> None:
    raise RuntimeError("limiter cache is down")


@pytest.mark.asyncio
async def test_a_turn_row_still_bills_when_freeing_its_parallel_slot_fails(local_model_cost_map: None):
    api: Final = _AgentsApi(
        turn_pages=((_turn("turn_1", CALL_STARTED_AT + 1, "completed", TURN_USAGE),),), session=_session()
    )
    clock: Final = _Clock()
    rows: Final = _SpendRows()

    await _relay_call(
        OpenAIAgentsTurnBiller(get=api.get, sleep=clock.sleep, clock=clock, release_request_slot=_failing_release),
        rows,
        EVENTS_URL,
        {"events": [INPUT_MESSAGE]},
        None,
        "8b0d2f4b-6d8f-4b0d-9f2b-4d6f8b0d2f01",
    )
    row: Final = await rows.only_row()

    assert row["total_tokens"] == TURN_USAGE["total_tokens"]
    assert row["spend"] == pytest.approx(_expected_cost(TURN_USAGE))
