import asyncio
import math
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Protocol, TypeAlias
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import (
    OPENAI_AGENTS_TURN_CLAIM_WINDOW_SECONDS,
    OPENAI_AGENTS_TURN_CLAIMS_MAX_SIZE,
    OPENAI_AGENTS_TURN_MAX_CONSECUTIVE_POLL_FAILURES,
    OPENAI_AGENTS_TURN_POLL_INITIAL_DELAY_SECONDS,
    OPENAI_AGENTS_TURN_POLL_MAX_DELAY_SECONDS,
    OPENAI_AGENTS_TURN_POLL_TIMEOUT_SECONDS,
    OPENAI_AGENTS_TURN_USAGE_SETTLE_SECONDS,
    OPENAI_AGENTS_TURNS_PAGE_LIMIT,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import (
    get_standard_logging_object_payload,
)
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.llms.custom_httpx.sse_lines import split_sse_lines
from litellm.proxy._types import PassThroughEndpointLoggingResultValues
from litellm.responses.utils import ResponseAPILoggingUtils
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.llms.openai import ResponseAPIUsage, ResponsesAPIResponse

from .transcribe_passthrough_logging_handler import PassThroughLogDispatch

_SESSION_ID: Final = re.compile(r"[A-Za-z0-9_-]+")
_TERMINAL_TURN_STATUSES: Final = frozenset({"completed", "failed", "cancelled"})
_TURN_STARTING_EVENT_TYPE: Final = "agent.session.input.message"


class AgentSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    service_tier: str | None = None


class _Session(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    agent: AgentSpec | None = None


class _SessionEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    session: _Session


class _RequestAgent(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent: AgentSpec


class _SessionCreateBody(BaseModel):
    model_config = ConfigDict(frozen=True)

    input: str | tuple[object, ...] | None = None


class _InputEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str


class _EventsBody(BaseModel):
    model_config = ConfigDict(frozen=True)

    events: tuple[_InputEvent, ...]


class _Turn(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    created_at: int
    status: str
    usage: ResponseAPIUsage | None = None


class _TurnsPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    data: tuple[_Turn, ...]


@dataclass(frozen=True, slots=True)
class AgentsTurnStart:
    base_url: str
    session_id: str
    started_at: int
    agent: AgentSpec | None


@dataclass(frozen=True, slots=True)
class DeferredAgentsRow:
    logging_obj: LiteLLMLoggingObj
    response_object: PassThroughEndpointLoggingResultValues
    result: str
    start_time: datetime
    end_time: datetime
    cache_hit: bool
    kwargs: Mapping[str, object]
    log: PassThroughLogDispatch


@dataclass(frozen=True, slots=True)
class _BilledTurn:
    turn: _Turn
    usage: ResponseAPIUsage


@dataclass(frozen=True, slots=True)
class _UnbilledTurn:
    reason: str


_TurnOutcome: TypeAlias = _BilledTurn | _UnbilledTurn


@dataclass(frozen=True, slots=True)
class _CreateRoute:
    prefix: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _SessionRoute:
    prefix: tuple[str, ...]
    session_id: str


_SessionsRoute: TypeAlias = _CreateRoute | _SessionRoute


@dataclass(frozen=True, slots=True)
class AgentsSessionsCall:
    origin: str
    route: _SessionsRoute


class AgentsApiGet(Protocol):
    def __call__(self, base_url: str, path: str) -> Awaitable[httpx.Response]: ...


class _TurnClaims(Protocol):
    def get_cache(self, key: str) -> object: ...

    def set_cache(self, key: str, value: object) -> None: ...


def _sessions_route(path: str) -> _SessionsRoute | None:
    match tuple(segment for segment in path.split("/") if segment):
        case (*prefix, "agents", "sessions"):
            return _CreateRoute(prefix=tuple(prefix))
        case (*prefix, "agents", "sessions", session_id, "events"):
            return _SessionRoute(prefix=tuple(prefix), session_id=session_id)
        case _:
            return None


def _has_input(request_body: Mapping[str, object]) -> bool:
    try:
        return bool(_SessionCreateBody.model_validate(request_body).input)
    except ValidationError:
        return False


def _has_input_message(request_body: Mapping[str, object]) -> bool:
    try:
        events: Final = _EventsBody.model_validate(request_body).events
    except ValidationError:
        return False
    return any(event.type == _TURN_STARTING_EVENT_TYPE for event in events)


def _request_agent(request_body: Mapping[str, object]) -> AgentSpec | None:
    try:
        return _RequestAgent.model_validate(request_body).agent
    except ValidationError:
        return None


def _session_from_body(response_body: object) -> _Session | None:
    try:
        return _Session.model_validate(response_body)
    except ValidationError:
        return None


def _session_from_sse_line(line: str) -> _Session | None:
    if not line.startswith("data:"):
        return None
    try:
        return _SessionEvent.model_validate_json(line.removeprefix("data:").strip()).session
    except ValidationError:
        return None


def _streamed_session(all_chunks: Sequence[str]) -> _Session | None:
    return next((session for line in all_chunks if (session := _session_from_sse_line(line)) is not None), None)


def _starts_turn(route: _SessionsRoute, request_body: Mapping[str, object]) -> bool:
    match route:
        case _CreateRoute():
            return _has_input(request_body)
        case _SessionRoute():
            return _has_input_message(request_body)


def _session_target(route: _SessionsRoute, session: _Session | None) -> _SessionRoute | None:
    match route:
        case _CreateRoute(prefix=prefix):
            return None if session is None else _SessionRoute(prefix=prefix, session_id=session.id)
        case _SessionRoute():
            return route


def openai_agents_sessions_call(url_route: str, custom_llm_provider: object) -> AgentsSessionsCall | None:
    from .openai_passthrough_logging_handler import is_openai_compatible_url

    if custom_llm_provider != "openai" or not is_openai_compatible_url(url_route):
        return None
    parsed_url: Final = urlparse(url_route)
    route: Final = _sessions_route(parsed_url.path)
    return (
        None if route is None else AgentsSessionsCall(origin=f"{parsed_url.scheme}://{parsed_url.netloc}", route=route)
    )


def _turn_start(
    call: AgentsSessionsCall,
    request_body: Mapping[str, object],
    relayed_session: Callable[[], _Session | None],
    start_time: datetime,
) -> AgentsTurnStart | None:
    if not _starts_turn(call.route, request_body):
        return None
    session: Final = relayed_session()
    target: Final = _session_target(call.route, session)
    if target is None or _SESSION_ID.fullmatch(target.session_id) is None:
        return None
    return AgentsTurnStart(
        base_url="/".join((call.origin, *target.prefix)),
        session_id=target.session_id,
        started_at=math.floor(start_time.timestamp()),
        agent=(session.agent if session is not None else None) or _request_agent(request_body),
    )


def agents_turn_start_from_response(
    call: AgentsSessionsCall,
    httpx_response: httpx.Response,
    request_body: Mapping[str, object],
    response_body: Mapping[str, object] | None,
    start_time: datetime,
) -> AgentsTurnStart | None:
    if not httpx_response.is_success or httpx_response.request.method != "POST":
        return None
    return _turn_start(call, request_body, lambda: _session_from_body(response_body), start_time)


def agents_turn_start_from_stream(
    call: AgentsSessionsCall,
    request_body: Mapping[str, object],
    raw_bytes: Sequence[bytes],
    start_time: datetime,
) -> AgentsTurnStart | None:
    return _turn_start(
        call,
        request_body,
        lambda: _streamed_session(split_sse_lines(b"".join(raw_bytes).decode("utf-8", errors="replace"))),
        start_time,
    )


async def live_agents_api_get(base_url: str, path: str) -> httpx.Response:
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        openai_passthrough_base_url,
        openai_passthrough_credential,
    )

    api_key: Final = await openai_passthrough_credential(openai_passthrough_base_url())
    auth_headers: Final[Mapping[str, str]] = (
        {} if api_key is None else {"authorization": f"Bearer {api_key}", "api-key": api_key}
    )
    return await get_async_httpx_client(llm_provider=httpxSpecialProvider.PassThroughEndpoint).client.get(
        f"{base_url}/{path}", headers={**auth_headers, "openai-beta": "agents=v1"}
    )


def _report_lost_row(task: "asyncio.Task[None]") -> None:
    if task.cancelled():
        verbose_proxy_logger.warning("OpenAI Agents passthrough: turn billing cancelled, the call logged no row")
        return
    error: Final = task.exception()
    if error is not None:
        verbose_proxy_logger.error("OpenAI Agents passthrough: turn billing failed, the call logged no row: %s", error)


def _turn_cost(agent: AgentSpec, turn: _Turn, usage: ResponseAPIUsage) -> float:
    try:
        return litellm.completion_cost(
            completion_response=ResponsesAPIResponse(
                id=turn.id,
                created_at=turn.created_at,
                model=agent.model,
                object="response",
                status=turn.status,
                output=[],
                usage=usage,
            ),
            model=agent.model,
            custom_llm_provider="openai",
            call_type="responses",
            service_tier=agent.service_tier,
        )
    except Exception as e:  # noqa: BLE001  # an unpriced model must still log the turn's tokens
        verbose_proxy_logger.warning(
            "OpenAI Agents passthrough: could not price turn %s for model %s: %s", turn.id, agent.model, e
        )
        return 0.0


class OpenAIAgentsTurnBiller:
    def __init__(
        self,
        get: AgentsApiGet,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._get: Final = get
        self._sleep: Final = sleep
        self._clock: Final = clock
        self._claims: Final[_TurnClaims] = InMemoryCache(
            max_size_in_memory=OPENAI_AGENTS_TURN_CLAIMS_MAX_SIZE,
            default_ttl=int(OPENAI_AGENTS_TURN_POLL_TIMEOUT_SECONDS),
        )
        self._tasks: Final[set[asyncio.Task[None]]] = set()  # mutable-ok: asyncio holds tasks weakly

    def schedule(self, turn_start: AgentsTurnStart, row: DeferredAgentsRow) -> "asyncio.Task[None]":
        task: Final = asyncio.create_task(self._bill(turn_start, row))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(_report_lost_row)
        return task

    async def _bill(self, turn_start: AgentsTurnStart, row: DeferredAgentsRow) -> None:
        outcome: Final = await self._await_billable_turn(turn_start, row.logging_obj.litellm_call_id)
        match outcome:
            case _BilledTurn(turn=turn, usage=usage):
                await self._log_billed(turn_start, row, turn, usage)
            case _UnbilledTurn(reason=reason):
                verbose_proxy_logger.warning(
                    "OpenAI Agents passthrough: logging session %s call without cost, %s",
                    turn_start.session_id,
                    reason,
                )
                await self._log(row, row.kwargs)

    async def _await_billable_turn(self, turn_start: AgentsTurnStart, claimant: str) -> _TurnOutcome:
        started: Final = self._clock()
        delay = OPENAI_AGENTS_TURN_POLL_INITIAL_DELAY_SECONDS  # rebind-ok: poll-loop backoff
        failures = 0  # rebind-ok: poll-loop state
        claimed_turn_id: str | None = None  # rebind-ok: poll-loop state
        settled_since: float | None = None  # rebind-ok: poll-loop state
        while self._clock() - started < OPENAI_AGENTS_TURN_POLL_TIMEOUT_SECONDS:
            await self._sleep(delay)
            delay = min(delay * 2, OPENAI_AGENTS_TURN_POLL_MAX_DELAY_SECONDS)
            page = await self._fetch_turns(turn_start)
            if page is None:
                failures += 1
                if failures >= OPENAI_AGENTS_TURN_MAX_CONSECUTIVE_POLL_FAILURES:
                    return _UnbilledTurn(reason=f"{failures} turn list polls failed in a row")
                continue
            failures = 0
            turn = (
                self._claim(page, turn_start, claimant)
                if claimed_turn_id is None
                else next((seen for seen in page.data if seen.id == claimed_turn_id), None)
            )
            if turn is None:
                if claimed_turn_id is None and self._clock() - started >= OPENAI_AGENTS_TURN_CLAIM_WINDOW_SECONDS:
                    return _UnbilledTurn(reason="no turn started by the call showed up")
                continue
            claimed_turn_id = turn.id
            if turn.status not in _TERMINAL_TURN_STATUSES:
                continue
            if turn.usage is not None:
                return _BilledTurn(turn=turn, usage=turn.usage)
            settled_since = settled_since if settled_since is not None else self._clock()
            if self._clock() - settled_since >= OPENAI_AGENTS_TURN_USAGE_SETTLE_SECONDS:
                return _UnbilledTurn(reason=f"turn {turn.id} ended without reporting usage")
        return _UnbilledTurn(reason=f"turn usage did not arrive within {OPENAI_AGENTS_TURN_POLL_TIMEOUT_SECONDS}s")

    def _claim(self, page: _TurnsPage, turn_start: AgentsTurnStart, claimant: str) -> _Turn | None:
        candidates: Final = sorted(
            (turn for turn in reversed(page.data) if turn.created_at >= turn_start.started_at),
            key=lambda turn: turn.created_at,
        )
        for turn in candidates:
            if self._claims.get_cache(turn.id) is None:
                self._claims.set_cache(turn.id, claimant)
                return turn
        return None

    async def _fetch(self, turn_start: AgentsTurnStart, path: str) -> bytes | None:
        try:
            response: Final = await self._get(turn_start.base_url, path)
        except httpx.HTTPError as e:
            verbose_proxy_logger.warning("OpenAI Agents passthrough: GET %s failed: %s", path, e)
            return None
        if not response.is_success:
            verbose_proxy_logger.warning("OpenAI Agents passthrough: GET %s returned %s", path, response.status_code)
            return None
        return response.content

    async def _fetch_turns(self, turn_start: AgentsTurnStart) -> _TurnsPage | None:
        content: Final = await self._fetch(
            turn_start,
            f"agents/sessions/{turn_start.session_id}/turns?order=desc&limit={OPENAI_AGENTS_TURNS_PAGE_LIMIT}",
        )
        if content is None:
            return None
        try:
            return _TurnsPage.model_validate_json(content)
        except ValidationError as e:
            verbose_proxy_logger.warning("OpenAI Agents passthrough: unreadable turns page: %s", e)
            return None

    async def _agent(self, turn_start: AgentsTurnStart) -> AgentSpec | None:
        if turn_start.agent is not None:
            return turn_start.agent
        content: Final = await self._fetch(turn_start, f"agents/sessions/{turn_start.session_id}")
        if content is None:
            return None
        try:
            return _Session.model_validate_json(content).agent
        except ValidationError as e:
            verbose_proxy_logger.warning("OpenAI Agents passthrough: unreadable session: %s", e)
            return None

    async def _log_billed(
        self, turn_start: AgentsTurnStart, row: DeferredAgentsRow, turn: _Turn, usage: ResponseAPIUsage
    ) -> None:
        agent: Final = await self._agent(turn_start)
        if agent is None:
            verbose_proxy_logger.warning(
                "OpenAI Agents passthrough: no agent model for session %s, logging turn %s without cost",
                turn_start.session_id,
                turn.id,
            )
        billed: Final[Mapping[str, object]] = {
            **({} if agent is None else {"model": agent.model}),
            "custom_llm_provider": "openai",
            "response_cost": 0.0 if agent is None else _turn_cost(agent, turn, usage),
            "combined_usage_object": ResponseAPILoggingUtils.transform_response_api_usage_to_chat_usage(usage),
        }
        row.logging_obj.model_call_details.update(billed)
        standard_logging_object: Final = get_standard_logging_object_payload(
            kwargs={**row.logging_obj.model_call_details, **row.kwargs, **billed},
            init_response_obj=row.response_object,
            start_time=row.start_time,
            end_time=row.end_time,
            logging_obj=row.logging_obj,
            status="success",
        )
        await self._log(row, {**row.kwargs, **billed, "standard_logging_object": standard_logging_object})

    @staticmethod
    async def _log(row: DeferredAgentsRow, kwargs: Mapping[str, object]) -> None:
        await row.log(
            logging_obj=row.logging_obj,
            standard_logging_response_object=row.response_object,
            result=row.result,
            start_time=row.start_time,
            end_time=row.end_time,
            cache_hit=row.cache_hit,
            **kwargs,
        )


openai_agents_turn_biller: Final = OpenAIAgentsTurnBiller(get=live_agents_api_get)
