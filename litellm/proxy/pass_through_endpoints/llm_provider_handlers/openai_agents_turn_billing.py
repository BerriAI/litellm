import asyncio
import itertools
import math
import re
import time
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Final, Protocol, TypeAlias, TypeVar
from urllib.parse import urlparse

import httpx
import openai
from pydantic import BaseModel, ConfigDict, ValidationError

import litellm
from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import (
    OPENAI_AGENTS_LIST_PAGE_LIMIT,
    OPENAI_AGENTS_SETTLED_FETCH_ATTEMPTS,
    OPENAI_AGENTS_TURN_CLAIM_WINDOW_SECONDS,
    OPENAI_AGENTS_TURN_CLAIMS_MAX_SIZE,
    OPENAI_AGENTS_TURN_CLOCK_SKEW_SECONDS,
    OPENAI_AGENTS_TURN_DRAIN_POLL_SHARE,
    OPENAI_AGENTS_TURN_POLL_INITIAL_DELAY_SECONDS,
    OPENAI_AGENTS_TURN_POLL_MAX_DELAY_SECONDS,
    OPENAI_AGENTS_TURN_POLL_TIMEOUT_SECONDS,
    PASS_THROUGH_HEADER_PREFIX,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import (
    get_standard_logging_object_payload,  # pyright: ignore[reportUnknownVariableType]  # kwargs is untyped upstream
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared factory leaves params untyped
)
from litellm.llms.custom_httpx.sse_lines import split_sse_lines
from litellm.llms.openai.common_utils import OpenAIError as LiteLLMOpenAIError
from litellm.passthrough.utils import BasePassthroughUtils
from litellm.proxy._types import PassThroughEndpointLoggingResultValues
from litellm.responses.utils import ResponseAPILoggingUtils
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.llms.openai import (
    InputTokensDetails,
    OutputTokensDetails,
    ResponseAPIUsage,
    ResponsesAPIResponse,
)

from .transcribe_passthrough_logging_handler import PassThroughLogDispatch

_SESSION_ID: Final = re.compile(r"[A-Za-z0-9_-]+")
_TERMINAL_TURN_STATUSES: Final = frozenset({"completed", "failed", "cancelled"})
_TURN_STARTING_EVENT_TYPE: Final = "agent.session.input.message"
_SCOPE_HEADERS: Final = frozenset({"openai-organization", "openai-project"})
_TURN_CLAIM_KEY_PREFIX: Final = "openai_agents_turn_claim:"
_RETRYABLE_STATUS_CODES: Final = frozenset({408, 409, 429})
_WEB_SEARCH_CALL: Final = "web_search_call"
_POLL_ERRORS: Final = (httpx.HTTPError, openai.OpenAIError, LiteLLMOpenAIError)


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


class _ContentPart(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str | None = None


class _Message(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str | None = None
    content: tuple[_ContentPart, ...] = ()

    @property
    def texts(self) -> Iterator[str]:
        return (part.text for part in self.content if part.text)


class _SessionCreateBody(BaseModel):
    model_config = ConfigDict(frozen=True)

    input: str | tuple[object, ...] | None = None


class _SessionCreateMessages(BaseModel):
    model_config = ConfigDict(frozen=True)

    input: str | tuple[_Message, ...] = ()


class _InputEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str


class _InputMessageEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str
    input: tuple[_Message, ...] = ()


class _MessageEventsBody(BaseModel):
    model_config = ConfigDict(frozen=True)

    events: tuple[_InputMessageEvent, ...]


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
    has_more: bool = False
    last_id: str | None = None


class _Subagent(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    closed_at: int | None = None


class _SubagentsPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    data: tuple[_Subagent, ...]
    has_more: bool = False
    last_id: str | None = None


class _Item(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str


class _ItemsPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    data: tuple[_Item, ...]
    has_more: bool = False
    last_id: str | None = None


class _MessageItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str
    role: str | None = None
    content: tuple[_ContentPart, ...] | str | None = None

    @property
    def user_texts(self) -> Iterator[str]:
        if self.type != "message" or self.role != "user" or not isinstance(self.content, tuple):
            return iter(())
        return (part.text for part in self.content if part.text)


class _MessageItemsPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    data: tuple[_MessageItem, ...]


_PageT = TypeVar("_PageT", _TurnsPage, _SubagentsPage, _ItemsPage, _MessageItemsPage, _Session)
_ResultT = TypeVar("_ResultT")


@dataclass(frozen=True, slots=True)
class AgentsTurnStart:
    base_url: str
    session_id: str
    started_at: int
    agent: AgentSpec | None
    scope_headers: Mapping[str, str]
    input_texts: frozenset[str] = frozenset()

    @property
    def earliest_turn_created_at(self) -> int:
        return self.started_at - OPENAI_AGENTS_TURN_CLOCK_SKEW_SECONDS if self.input_texts else self.started_at


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
class _ClaimedTurn:
    turn: _Turn
    turns_path: str

    @property
    def settled(self) -> bool:
        return self.turn.status in _TERMINAL_TURN_STATUSES and self.turn.usage is not None


@dataclass(frozen=True, slots=True)
class _Claimed:
    root: _ClaimedTurn | None = None
    subagent_turns: tuple[_ClaimedTurn, ...] = ()

    @property
    def turns(self) -> tuple[_ClaimedTurn, ...]:
        return self.subagent_turns if self.root is None else (self.root, *self.subagent_turns)

    @property
    def settled(self) -> bool:
        return self.root is not None and all(claimed_turn.settled for claimed_turn in self.turns)


_NOTHING_CLAIMED: Final = _Claimed()


@dataclass(frozen=True, slots=True)
class _FetchFailed:
    reason: str
    retryable: bool


@dataclass(frozen=True, slots=True)
class _Polled:
    claimed: _Claimed
    failure: _FetchFailed | None = None


@dataclass(frozen=True, slots=True)
class _BilledTurn:
    turn: _Turn
    usage: ResponseAPIUsage
    items_path: str


@dataclass(frozen=True, slots=True)
class _BilledTurns:
    turns: tuple[_BilledTurn, ...]
    unsettled_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _UnbilledTurn:
    reason: str


_UNREAD_AT_SHUTDOWN: Final = _UnbilledTurn(reason="the proxy shut down before its turns were read")


_TurnOutcome: TypeAlias = _BilledTurns | _UnbilledTurn


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
    def __call__(self, base_url: str, path: str, scope_headers: Mapping[str, str]) -> Awaitable[httpx.Response]: ...


class TurnClaimStore(Protocol):
    async def async_set_cache(self, key: str, value: object, *, nx: bool, ttl: int) -> object: ...

    async def async_get_cache(self, key: str) -> object: ...


class TurnClaims:
    def __init__(self, shared_store: Callable[[], TurnClaimStore | None]) -> None:
        self._shared_store: Final = shared_store
        self._local: Final = InMemoryCache(
            max_size_in_memory=OPENAI_AGENTS_TURN_CLAIMS_MAX_SIZE,
            default_ttl=int(OPENAI_AGENTS_TURN_POLL_TIMEOUT_SECONDS),
        )

    async def claim(self, turn_id: str, claimant: str) -> bool:
        if self._held_here(turn_id):
            return False
        self._hold_here(turn_id, claimant)
        return await self._shared_owner(turn_id, claimant) == claimant

    async def taken(self, turn_id: str) -> bool:
        if self._held_here(turn_id):
            return True
        claimant: Final = await self._shared_claimant(turn_id)
        if claimant is None:
            return False
        self._hold_here(turn_id, claimant)
        return True

    def _held_here(self, turn_id: str) -> bool:
        return self._local.get_cache(turn_id) is not None

    def _hold_here(self, turn_id: str, claimant: object) -> None:
        self._local.set_cache(turn_id, claimant)

    @with_service_target("openai_agents_turn_claims")
    async def _shared_claimant(self, turn_id: str) -> object:
        store: Final = self._shared_store()
        if store is None:
            return None
        try:
            return await store.async_get_cache(_claim_key(turn_id))
        except Exception as e:  # noqa: BLE001  # a Redis outage leaves the turn to claim(), which falls back to this worker
            verbose_proxy_logger.warning("OpenAI Agents passthrough: reading a shared turn claim failed: %s", e)
            return None

    @with_service_target("openai_agents_turn_claims")
    async def _shared_owner(self, turn_id: str, claimant: str) -> object:
        store: Final = self._shared_store()
        if store is None:
            return claimant
        key: Final = _claim_key(turn_id)
        try:
            _ = await store.async_set_cache(key, claimant, nx=True, ttl=int(OPENAI_AGENTS_TURN_POLL_TIMEOUT_SECONDS))
            stored: Final = await store.async_get_cache(key)
        except Exception as e:  # noqa: BLE001  # a Redis outage keeps the claim in this worker, as DualCache's writes do
            verbose_proxy_logger.warning("OpenAI Agents passthrough: shared turn claim failed, claiming locally: %s", e)
            return claimant
        return claimant if stored is None else stored


def _claim_key(turn_id: str) -> str:
    return f"{_TURN_CLAIM_KEY_PREFIX}{turn_id}"


def _coordination_redis() -> TurnClaimStore | None:
    from litellm.proxy.proxy_server import redis_usage_cache

    return redis_usage_cache


async def _release_request_parallel_slot(logging_obj: LiteLLMLoggingObj) -> None:
    from litellm.proxy.proxy_server import proxy_logging_obj

    await proxy_logging_obj.arelease_slot_before_logging(logging_obj)


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


def _create_input_texts(request_body: Mapping[str, object]) -> frozenset[str]:
    try:
        create_input: Final = _SessionCreateMessages.model_validate(request_body).input
    except ValidationError:
        return frozenset()
    if isinstance(create_input, str):
        return frozenset((create_input,)) if create_input else frozenset()
    return frozenset(itertools.chain.from_iterable(message.texts for message in create_input))


def _events_input_texts(request_body: Mapping[str, object]) -> frozenset[str]:
    try:
        events: Final = _MessageEventsBody.model_validate(request_body).events
    except ValidationError:
        return frozenset()
    messages: Final = itertools.chain.from_iterable(
        event.input for event in events if event.type == _TURN_STARTING_EVENT_TYPE
    )
    return frozenset(itertools.chain.from_iterable(message.texts for message in messages))


def _input_texts(route: _SessionsRoute, request_body: Mapping[str, object]) -> frozenset[str]:
    match route:
        case _CreateRoute():
            return _create_input_texts(request_body)
        case _SessionRoute():
            return _events_input_texts(request_body)


def _scope_headers(request_headers: Mapping[str, object] | None) -> Mapping[str, str]:
    forwarded: Final = BasePassthroughUtils.pass_through_prefixed_headers(
        {name: value for name, value in (request_headers or {}).items() if isinstance(value, str)}
    )
    named: Final = ((name[len(PASS_THROUGH_HEADER_PREFIX) :].lower(), value) for name, value in forwarded.items())
    return MappingProxyType({name: value for name, value in named if name in _SCOPE_HEADERS})


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
    request_headers: Mapping[str, object] | None,
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
        agent=session.agent if session is not None else None,
        scope_headers=_scope_headers(request_headers),
        input_texts=_input_texts(call.route, request_body),
    )


def agents_turn_start_from_response(
    call: AgentsSessionsCall,
    httpx_response: httpx.Response,
    request_body: Mapping[str, object],
    request_headers: Mapping[str, object] | None,
    response_body: Mapping[str, object] | None,
    start_time: datetime,
) -> AgentsTurnStart | None:
    if not httpx_response.is_success or httpx_response.request.method != "POST":
        return None
    return _turn_start(call, request_body, request_headers, lambda: _session_from_body(response_body), start_time)


def agents_turn_start_from_stream(
    call: AgentsSessionsCall,
    request_body: Mapping[str, object],
    request_headers: Mapping[str, object] | None,
    raw_bytes: Sequence[bytes],
    start_time: datetime,
) -> AgentsTurnStart | None:
    return _turn_start(
        call,
        request_body,
        request_headers,
        lambda: _streamed_session(split_sse_lines(b"".join(raw_bytes).decode("utf-8", errors="replace"))),
        start_time,
    )


async def live_agents_api_get(base_url: str, path: str, scope_headers: Mapping[str, str]) -> httpx.Response:
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
        openai_api_key_headers,
        openai_passthrough_base_url,
        openai_passthrough_credential,
    )

    auth_headers: Final = openai_api_key_headers(await openai_passthrough_credential(openai_passthrough_base_url()))
    return await get_async_httpx_client(llm_provider=httpxSpecialProvider.PassThroughEndpoint).client.get(
        f"{base_url}/{path}",
        headers={**scope_headers, **auth_headers, "openai-beta": "agents=v1"},
        follow_redirects=False,
    )


def _report_lost_row(task: "asyncio.Task[None]") -> None:
    if task.cancelled():
        verbose_proxy_logger.warning("OpenAI Agents passthrough: turn billing cancelled, the call logged no row")
        return
    error: Final = task.exception()
    if error is not None:
        verbose_proxy_logger.error("OpenAI Agents passthrough: turn billing failed, the call logged no row: %s", error)


async def _unless_set(work: Awaitable[_ResultT], interrupt: asyncio.Event) -> _ResultT | None:
    working: Final = asyncio.ensure_future(work)
    interrupted: Final = asyncio.ensure_future(interrupt.wait())
    try:
        done, _ = await asyncio.wait((working, interrupted), return_when=asyncio.FIRST_COMPLETED)
    finally:
        _ = working.cancel()
        _ = interrupted.cancel()
    return working.result() if working in done else None


def _is_retryable(status_code: int) -> bool:
    return status_code >= 500 or status_code in _RETRYABLE_STATUS_CODES


def _billed_turns(claimed: _Claimed) -> tuple[_BilledTurn, ...]:
    return tuple(
        _BilledTurn(
            turn=claimed_turn.turn, usage=usage, items_path=f"{claimed_turn.turns_path}/{claimed_turn.turn.id}/items"
        )
        for claimed_turn in claimed.turns
        if (usage := claimed_turn.turn.usage) is not None
    )


def _bill_reported(claimed: _Claimed, reason: str) -> _TurnOutcome:
    reported: Final = _billed_turns(claimed)
    return _BilledTurns(turns=reported, unsettled_reason=reason) if reported else _UnbilledTurn(reason=reason)


def _listed_turns(turns_paths: Sequence[str], pages: Sequence[_TurnsPage | _FetchFailed]) -> Iterator[_ClaimedTurn]:
    for turns_path, page in zip(turns_paths, pages, strict=True):
        if isinstance(page, _TurnsPage):
            yield from (_ClaimedTurn(turn=turn, turns_path=turns_path) for turn in page.data)


def _cached_tokens(usage: ResponseAPIUsage) -> int:
    return 0 if usage.input_tokens_details is None else usage.input_tokens_details.cached_tokens


def _reasoning_tokens(usage: ResponseAPIUsage) -> int:
    return 0 if usage.output_tokens_details is None else usage.output_tokens_details.reasoning_tokens or 0


def _combined_usage(usages: Sequence[ResponseAPIUsage]) -> ResponseAPIUsage:
    if len(usages) == 1:
        return usages[0]
    return ResponseAPIUsage(
        input_tokens=sum(usage.input_tokens for usage in usages),
        input_tokens_details=InputTokensDetails(cached_tokens=sum(_cached_tokens(usage) for usage in usages)),
        output_tokens=sum(usage.output_tokens for usage in usages),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=sum(_reasoning_tokens(usage) for usage in usages)),
        total_tokens=sum(usage.total_tokens for usage in usages),
    )


def _turn_cost(agent: AgentSpec, billed_turn: _BilledTurn, web_searches: int) -> float:
    try:
        return litellm.completion_cost(
            completion_response=ResponsesAPIResponse(
                id=billed_turn.turn.id,
                created_at=billed_turn.turn.created_at,
                model=agent.model,
                object="response",
                status=billed_turn.turn.status,
                output=[
                    {"type": _WEB_SEARCH_CALL, "id": f"{billed_turn.turn.id}_{index}", "status": "completed"}
                    for index in range(web_searches)
                ],
                usage=billed_turn.usage,
            ),
            model=agent.model,
            custom_llm_provider="openai",
            call_type="responses",
            service_tier=agent.service_tier,
        )
    except Exception as e:  # noqa: BLE001  # an unpriced model must still log the turn's tokens
        verbose_proxy_logger.warning(
            "OpenAI Agents passthrough: could not price turn %s for model %s: %s", billed_turn.turn.id, agent.model, e
        )
        return 0.0


class OpenAIAgentsTurnBiller:
    def __init__(
        self,
        get: AgentsApiGet,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        claims: TurnClaims | None = None,
        release_request_slot: Callable[[LiteLLMLoggingObj], Awaitable[None]] = _release_request_parallel_slot,
    ) -> None:
        self._get: Final = get
        self._sleep: Final = sleep
        self._clock: Final = clock
        self._claims: Final = claims if claims is not None else TurnClaims(shared_store=lambda: None)
        self._release_request_slot: Final = release_request_slot
        self._tasks: Final[set[asyncio.Task[None]]] = set()  # mutable-ok: asyncio holds tasks weakly
        self._stopping: Final = asyncio.Event()
        self._abandoning: Final = asyncio.Event()

    def schedule(self, turn_start: AgentsTurnStart, row: DeferredAgentsRow) -> "asyncio.Task[None]":
        task: Final = asyncio.create_task(self._bill(turn_start, row))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(_report_lost_row)
        return task

    async def drain(self, timeout: float) -> None:
        self._stopping.set()
        waiting: Final = tuple(self._tasks)
        if not waiting:
            return
        try:
            _, still_polling = await asyncio.wait(waiting, timeout=timeout * OPENAI_AGENTS_TURN_DRAIN_POLL_SHARE)
        finally:
            self._abandoning.set()
        if still_polling:
            _ = await asyncio.wait(still_polling, timeout=timeout * (1 - OPENAI_AGENTS_TURN_DRAIN_POLL_SHARE))

    async def _bill(self, turn_start: AgentsTurnStart, row: DeferredAgentsRow) -> None:
        await self._release_slot(turn_start, row)
        costed: Final = await _unless_set(self._costed_kwargs(turn_start, row), self._abandoning)
        await self._log(
            row, costed if costed is not None else await self._row_kwargs(turn_start, row, _UNREAD_AT_SHUTDOWN)
        )

    async def _costed_kwargs(self, turn_start: AgentsTurnStart, row: DeferredAgentsRow) -> Mapping[str, object]:
        outcome: Final = await self._turn_outcome(turn_start, row.logging_obj.litellm_call_id)
        return await self._row_kwargs(turn_start, row, outcome)

    async def _release_slot(self, turn_start: AgentsTurnStart, row: DeferredAgentsRow) -> None:
        try:
            await self._release_request_slot(row.logging_obj)
        except Exception as e:  # noqa: BLE001  # a limiter outage must not cost the call its spend row
            verbose_proxy_logger.warning(
                "OpenAI Agents passthrough: could not free the parallel request slot of session %s early: %r",
                turn_start.session_id,
                e,
            )

    async def _row_kwargs(
        self, turn_start: AgentsTurnStart, row: DeferredAgentsRow, outcome: _TurnOutcome
    ) -> Mapping[str, object]:
        try:
            return await self._outcome_kwargs(turn_start, row, outcome)
        except Exception as e:  # noqa: BLE001  # the call already succeeded upstream, so its row logs even when costing breaks
            verbose_proxy_logger.warning(
                "OpenAI Agents passthrough: logging session %s call without cost, costing its turns failed: %r",
                turn_start.session_id,
                e,
            )
            return row.kwargs

    async def _outcome_kwargs(
        self, turn_start: AgentsTurnStart, row: DeferredAgentsRow, outcome: _TurnOutcome
    ) -> Mapping[str, object]:
        match outcome:
            case _BilledTurns(turns=turns, unsettled_reason=None):
                return await self._billed_kwargs(turn_start, row, turns)
            case _BilledTurns(turns=turns, unsettled_reason=reason):
                verbose_proxy_logger.warning(
                    "OpenAI Agents passthrough: billing session %s with the usage its turns reported so far, %s",
                    turn_start.session_id,
                    reason,
                )
                return await self._billed_kwargs(turn_start, row, turns)
            case _UnbilledTurn(reason=reason):
                verbose_proxy_logger.warning(
                    "OpenAI Agents passthrough: logging session %s call without cost, %s",
                    turn_start.session_id,
                    reason,
                )
                return row.kwargs

    async def _turn_outcome(self, turn_start: AgentsTurnStart, claimant: str) -> _TurnOutcome:
        try:
            return await self._await_billable_turns(turn_start, claimant, self._clock())
        except Exception as e:  # noqa: BLE001  # the call already succeeded upstream, so its row logs even when polling breaks
            return _UnbilledTurn(reason=f"polling its turns failed: {e!r}")

    async def _await_billable_turns(
        self,
        turn_start: AgentsTurnStart,
        claimant: str,
        started: float,
        delay: float = OPENAI_AGENTS_TURN_POLL_INITIAL_DELAY_SECONDS,
        claimed: _Claimed = _NOTHING_CLAIMED,
    ) -> _TurnOutcome:
        await self._wait(delay)
        polled: Final = await self._poll(turn_start, claimant, claimed)
        outcome: Final = self._outcome(polled, self._clock() - started)
        if outcome is not None:
            return outcome
        return await self._await_billable_turns(
            turn_start,
            claimant,
            started,
            min(delay * 2, OPENAI_AGENTS_TURN_POLL_MAX_DELAY_SECONDS),
            polled.claimed,
        )

    async def _wait(self, delay: float) -> None:
        if not self._stopping.is_set():
            _ = await _unless_set(self._sleep(delay), self._stopping)

    def _outcome(self, polled: _Polled, elapsed: float) -> _TurnOutcome | None:
        failure: Final = polled.failure
        claimed: Final = polled.claimed
        if failure is not None and not failure.retryable:
            return _bill_reported(claimed, failure.reason)
        if failure is None and claimed.settled:
            return _BilledTurns(turns=_billed_turns(claimed))
        if failure is None and claimed.root is None and elapsed >= OPENAI_AGENTS_TURN_CLAIM_WINDOW_SECONDS:
            return _UnbilledTurn(reason="no turn started by the call showed up")
        if self._stopping.is_set():
            return _bill_reported(claimed, "the proxy is shutting down")
        if elapsed >= OPENAI_AGENTS_TURN_POLL_TIMEOUT_SECONDS:
            return _bill_reported(
                claimed, f"its turns did not settle within {OPENAI_AGENTS_TURN_POLL_TIMEOUT_SECONDS:g}s"
            )
        return None

    async def _poll(self, turn_start: AgentsTurnStart, claimant: str, claimed: _Claimed) -> _Polled:
        root: Final = await self._refreshed_root(turn_start, claimant, claimed.root)
        if isinstance(root, _FetchFailed):
            return _Polled(claimed=claimed, failure=root)
        if root is None or not root.settled:
            return _Polled(claimed=_Claimed(root=root))
        subagent_turns: Final = await self._refreshed_subagent_turns(turn_start, claimant, root, claimed.subagent_turns)
        if isinstance(subagent_turns, _FetchFailed):
            return _Polled(claimed=_Claimed(root=root, subagent_turns=claimed.subagent_turns), failure=subagent_turns)
        return _Polled(claimed=_Claimed(root=root, subagent_turns=subagent_turns))

    async def _refreshed_root(
        self, turn_start: AgentsTurnStart, claimant: str, root: _ClaimedTurn | None
    ) -> _ClaimedTurn | _FetchFailed | None:
        if root is not None and root.settled:
            return root
        turns_path: Final = f"agents/sessions/{turn_start.session_id}/turns"
        since: Final = turn_start.earliest_turn_created_at if root is None else root.turn.created_at
        page: Final = await self._fetch_turns(turn_start, turns_path, since)
        if isinstance(page, _FetchFailed):
            return page
        if root is None:
            return await self._claim(page, turn_start, claimant, turns_path)
        return next(
            (_ClaimedTurn(turn=turn, turns_path=turns_path) for turn in page.data if turn.id == root.turn.id), root
        )

    async def _claim(
        self, page: _TurnsPage, turn_start: AgentsTurnStart, claimant: str, turns_path: str
    ) -> _ClaimedTurn | _FetchFailed | None:
        candidates: Final = sorted(
            (turn for turn in reversed(page.data) if turn.created_at >= turn_start.earliest_turn_created_at),
            key=lambda turn: turn.created_at,
        )
        for turn in candidates:
            if await self._claims.taken(turn.id):
                continue
            if isinstance(started := await self._started_by_call(turn_start, turns_path, turn), _FetchFailed):
                return started
            if started and await self._claims.claim(turn.id, claimant):
                return _ClaimedTurn(turn=turn, turns_path=turns_path)
        return None

    async def _started_by_call(self, turn_start: AgentsTurnStart, turns_path: str, turn: _Turn) -> bool | _FetchFailed:
        if not turn_start.input_texts:
            return True
        items: Final = await self._fetch_page(
            turn_start,
            f"{turns_path}/{turn.id}/items?order=asc&limit={OPENAI_AGENTS_LIST_PAGE_LIMIT}",
            _MessageItemsPage,
        )
        if isinstance(items, _FetchFailed):
            return items if items.retryable else False
        turn_texts: Final = itertools.chain.from_iterable(item.user_texts for item in items.data)
        return not turn_start.input_texts.isdisjoint(turn_texts)

    async def _refreshed_subagent_turns(
        self, turn_start: AgentsTurnStart, claimant: str, root: _ClaimedTurn, known: tuple[_ClaimedTurn, ...]
    ) -> tuple[_ClaimedTurn, ...] | _FetchFailed:
        session_path: Final = f"agents/sessions/{turn_start.session_id}"
        subagents: Final = await self._subagents(turn_start, session_path)
        if isinstance(subagents, _FetchFailed):
            return subagents
        root_created_at: Final = root.turn.created_at
        open_during_turn: Final = (
            subagent for subagent in subagents if subagent.closed_at is None or subagent.closed_at >= root_created_at
        )
        turns_paths: Final = tuple(f"{session_path}/subagents/{subagent.id}/turns" for subagent in open_during_turn)
        pages: Final = await asyncio.gather(
            *(self._fetch_turns(turn_start, turns_path, root_created_at) for turns_path in turns_paths)
        )
        failure: Final = next((page for page in pages if isinstance(page, _FetchFailed)), None)
        if failure is not None:
            return failure
        listed: Final = tuple(_listed_turns(turns_paths, pages))
        if not listed:
            return known
        next_root_created_at: Final = await self._next_root_created_at(turn_start, root)
        if isinstance(next_root_created_at, _FetchFailed):
            return next_root_created_at
        latest: Final = MappingProxyType({claimed_turn.turn.id: claimed_turn for claimed_turn in listed})
        known_ids: Final = frozenset(claimed_turn.turn.id for claimed_turn in known)
        claimed_now: Final = tuple(
            [
                claimed_turn
                for claimed_turn in listed
                if claimed_turn.turn.id not in known_ids
                and root_created_at <= claimed_turn.turn.created_at < next_root_created_at
                and await self._claims.claim(claimed_turn.turn.id, claimant)
            ]
        )
        return (*(latest.get(claimed_turn.turn.id, claimed_turn) for claimed_turn in known), *claimed_now)

    async def _subagents(
        self, turn_start: AgentsTurnStart, session_path: str, after: str | None = None
    ) -> tuple[_Subagent, ...] | _FetchFailed:
        cursor: Final = "" if after is None else f"&after={after}"
        page: Final = await self._fetch_page(
            turn_start,
            f"{session_path}/subagents?order=desc&limit={OPENAI_AGENTS_LIST_PAGE_LIMIT}{cursor}",
            _SubagentsPage,
        )
        if isinstance(page, _FetchFailed):
            return page
        if not page.has_more or page.last_id is None:
            return page.data
        older: Final = await self._subagents(turn_start, session_path, page.last_id)
        return older if isinstance(older, _FetchFailed) else (*page.data, *older)

    async def _next_root_created_at(self, turn_start: AgentsTurnStart, root: _ClaimedTurn) -> float | _FetchFailed:
        roots: Final = await self._fetch_turns(turn_start, root.turns_path, root.turn.created_at)
        if isinstance(roots, _FetchFailed):
            return roots
        return min((turn.created_at for turn in roots.data if turn.created_at > root.turn.created_at), default=math.inf)

    async def _fetch(self, turn_start: AgentsTurnStart, path: str) -> bytes | _FetchFailed:
        try:
            response: Final = await self._get(turn_start.base_url, path, turn_start.scope_headers)
        except _POLL_ERRORS as e:
            verbose_proxy_logger.warning("OpenAI Agents passthrough: GET %s failed: %s", path, e)
            return _FetchFailed(reason=f"GET {path} failed: {e}", retryable=True)
        if not response.is_success:
            verbose_proxy_logger.warning("OpenAI Agents passthrough: GET %s returned %s", path, response.status_code)
            return _FetchFailed(
                reason=f"GET {path} returned {response.status_code}", retryable=_is_retryable(response.status_code)
            )
        return response.content

    async def _fetch_page(
        self, turn_start: AgentsTurnStart, path: str, page_type: type[_PageT]
    ) -> _PageT | _FetchFailed:
        content: Final = await self._fetch(turn_start, path)
        if isinstance(content, _FetchFailed):
            return content
        try:
            return page_type.model_validate_json(content)
        except ValidationError as e:
            verbose_proxy_logger.warning(
                "OpenAI Agents passthrough: GET %s returned an unreadable body: %s",
                path,
                tuple((error["type"], error["loc"]) for error in e.errors(include_url=False)),
            )
            return _FetchFailed(reason=f"GET {path} returned an unreadable body", retryable=False)

    async def _fetch_settled_page(
        self,
        turn_start: AgentsTurnStart,
        path: str,
        page_type: type[_PageT],
        attempts_left: int = OPENAI_AGENTS_SETTLED_FETCH_ATTEMPTS,
        delay: float = OPENAI_AGENTS_TURN_POLL_INITIAL_DELAY_SECONDS,
    ) -> _PageT | _FetchFailed:
        page: Final = await self._fetch_page(turn_start, path, page_type)
        if not isinstance(page, _FetchFailed) or not page.retryable:
            return page
        if attempts_left <= 1 or self._stopping.is_set():
            return page
        await self._wait(delay)
        return await self._fetch_settled_page(
            turn_start, path, page_type, attempts_left - 1, min(delay * 2, OPENAI_AGENTS_TURN_POLL_MAX_DELAY_SECONDS)
        )

    async def _fetch_turns(
        self, turn_start: AgentsTurnStart, turns_path: str, since: int, after: str | None = None
    ) -> _TurnsPage | _FetchFailed:
        cursor: Final = "" if after is None else f"&after={after}"
        page: Final = await self._fetch_page(
            turn_start, f"{turns_path}?order=desc&limit={OPENAI_AGENTS_LIST_PAGE_LIMIT}{cursor}", _TurnsPage
        )
        if isinstance(page, _FetchFailed):
            return page
        if not page.has_more or page.last_id is None or any(turn.created_at < since for turn in page.data):
            return page
        older: Final = await self._fetch_turns(turn_start, turns_path, since, page.last_id)
        return older if isinstance(older, _FetchFailed) else _TurnsPage(data=(*page.data, *older.data))

    async def _web_search_calls(self, turn_start: AgentsTurnStart, items_path: str, after: str | None = None) -> int:
        cursor: Final = "" if after is None else f"&after={after}"
        page: Final = await self._fetch_settled_page(
            turn_start, f"{items_path}?limit={OPENAI_AGENTS_LIST_PAGE_LIMIT}{cursor}", _ItemsPage
        )
        if isinstance(page, _FetchFailed):
            verbose_proxy_logger.warning(
                "OpenAI Agents passthrough: billing session %s without the web searches of %s, %s",
                turn_start.session_id,
                items_path,
                page.reason,
            )
            return 0
        searches: Final = sum(1 for item in page.data if item.type == _WEB_SEARCH_CALL)
        if not page.has_more or page.last_id is None:
            return searches
        return searches + await self._web_search_calls(turn_start, items_path, page.last_id)

    async def _agent(self, turn_start: AgentsTurnStart) -> AgentSpec | None:
        if turn_start.agent is not None:
            return turn_start.agent
        session: Final = await self._fetch_settled_page(
            turn_start, f"agents/sessions/{turn_start.session_id}", _Session
        )
        return None if isinstance(session, _FetchFailed) else session.agent

    async def _response_cost(
        self, turn_start: AgentsTurnStart, agent: AgentSpec, turns: tuple[_BilledTurn, ...]
    ) -> float:
        return sum(
            [
                _turn_cost(agent, billed_turn, await self._web_search_calls(turn_start, billed_turn.items_path))
                for billed_turn in turns
            ]
        )

    async def _billed_kwargs(
        self, turn_start: AgentsTurnStart, row: DeferredAgentsRow, turns: tuple[_BilledTurn, ...]
    ) -> Mapping[str, object]:
        agent: Final = await self._agent(turn_start)
        if agent is None:
            verbose_proxy_logger.warning(
                "OpenAI Agents passthrough: no agent model for session %s, logging its turns without cost",
                turn_start.session_id,
            )
        billed: Final[Mapping[str, object]] = {
            **({} if agent is None else {"model": agent.model}),
            "custom_llm_provider": "openai",
            "response_cost": 0.0 if agent is None else await self._response_cost(turn_start, agent, turns),
            "combined_usage_object": ResponseAPILoggingUtils.transform_response_api_usage_to_chat_usage(
                _combined_usage(tuple(billed_turn.usage for billed_turn in turns))
            ),
        }
        row.logging_obj.model_call_details.update(billed)
        standard_logging_object: Final = get_standard_logging_object_payload(
            kwargs={**row.kwargs, **row.logging_obj.model_call_details, **billed},
            init_response_obj=row.response_object,
            start_time=row.start_time,
            end_time=row.end_time,
            logging_obj=row.logging_obj,
            status="success",
        )
        return {**row.kwargs, **billed, "standard_logging_object": standard_logging_object}

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


openai_agents_turn_biller: Final = OpenAIAgentsTurnBiller(
    get=live_agents_api_get, claims=TurnClaims(shared_store=_coordination_redis)
)
