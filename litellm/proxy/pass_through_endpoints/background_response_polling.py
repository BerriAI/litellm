import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final, TypeAlias
from urllib.parse import quote

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.interactions.background_cost_polling import PollSchedule, poll_intervals
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared client factory has an untyped params dict
)
from litellm.types.llms.custom_http import httpxSpecialProvider

from .background_response_creator import BackgroundResponseCreator, BackgroundResponseCreatorStore

BACKGROUND_RESPONSE_POLL_SCHEDULE: Final = PollSchedule(
    initial_interval_seconds=5.0, max_interval_seconds=60.0, timeout_seconds=7800.0
)

POLLABLE_BACKGROUND_RESPONSE_STATUSES: Final = frozenset({"queued", "in_progress"})

_UNSAFE_RESPONSE_ID_PARTS: Final = ("/", "\\", "#", "?", "..")

_HEADERS_NOT_SENT_ON_POLL: Final = frozenset(
    {"content-length", "content-type", "content-encoding", "transfer-encoding", "host"}
)

_RETRYABLE_STATUS_CODES: Final = frozenset({408, 409, 429})

_RESPONSE_BODY: Final = TypeAdapter(dict[str, object])


def is_pollable_response_id(response_id: str) -> bool:
    return bool(response_id) and not any(part in response_id for part in _UNSAFE_RESPONSE_ID_PARTS)


def response_url(collection_url: httpx.URL, response_id: str) -> httpx.URL:
    return collection_url.copy_with(path=f"{collection_url.path.rstrip('/')}/{quote(response_id, safe='')}")


@dataclass(frozen=True, slots=True)
class BackgroundResponsePollRequest:
    url: httpx.URL
    headers: tuple[tuple[str, str], ...]


def poll_request_for_create(create_request: httpx.Request, response_id: str) -> BackgroundResponsePollRequest:
    return BackgroundResponsePollRequest(
        url=response_url(create_request.url, response_id),
        headers=tuple(
            (name, value)
            for name, value in create_request.headers.multi_items()
            if name.lower() not in _HEADERS_NOT_SENT_ON_POLL
        ),
    )


@dataclass(frozen=True, slots=True)
class Fetched:
    body: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class RetryLater:
    reason: str


@dataclass(frozen=True, slots=True)
class StopPolling:
    status_code: int


FetchOutcome: TypeAlias = Fetched | RetryLater | StopPolling

FetchBackgroundResponse: TypeAlias = Callable[[BackgroundResponsePollRequest], Awaitable[FetchOutcome]]

SettlePolledResponse: TypeAlias = Callable[[BackgroundResponseCreator, Mapping[str, object]], Awaitable[bool]]


def _fetch_outcome(response: httpx.Response) -> FetchOutcome:
    if response.status_code in _RETRYABLE_STATUS_CODES or response.status_code >= 500:
        return RetryLater(reason=f"the provider answered {response.status_code}")
    if not response.is_success:
        return StopPolling(status_code=response.status_code)
    try:
        return Fetched(body=_RESPONSE_BODY.validate_json(response.content))
    except ValidationError:
        return RetryLater(reason="the provider answered with a body that is not a JSON object")


def _pass_through_client() -> httpx.AsyncClient:
    return get_async_httpx_client(llm_provider=httpxSpecialProvider.PassThroughEndpoint).client


@dataclass(frozen=True, slots=True)
class HttpxBackgroundResponseFetch:
    client: Callable[[], httpx.AsyncClient] = _pass_through_client

    async def __call__(self, request: BackgroundResponsePollRequest) -> FetchOutcome:
        try:
            response: Final = await self.client().get(request.url, headers=list(request.headers))
        except httpx.HTTPError as e:
            return RetryLater(reason=f"{type(e).__name__}: {e}")
        return _fetch_outcome(response)


async def poll_background_response(
    creator: BackgroundResponseCreator,
    request: BackgroundResponsePollRequest,
    settle: SettlePolledResponse,
    fetch: FetchBackgroundResponse,
    schedule: PollSchedule,
) -> None:
    intervals: Final = poll_intervals(
        schedule.initial_interval_seconds, schedule.max_interval_seconds, schedule.timeout_seconds
    )
    for interval in intervals:
        await asyncio.sleep(interval)
        match await fetch(request):
            case Fetched(body=body) if body.get("status") in POLLABLE_BACKGROUND_RESPONSE_STATUSES:
                continue
            case Fetched(body=body):
                if await settle(creator, body):
                    return
                verbose_proxy_logger.debug(
                    "Could not record the settlement of background response %s, will retry", creator.response_id
                )
            case RetryLater(reason=reason):
                verbose_proxy_logger.debug(
                    "Polling background response %s failed, will retry: %s", creator.response_id, reason
                )
            case StopPolling(status_code=status_code):
                verbose_proxy_logger.warning(
                    "Stopped polling background response %s: the provider answered %s, so a later read through "
                    "the proxy bills the key that created it",
                    creator.response_id,
                    status_code,
                )
                return
    verbose_proxy_logger.warning(
        "Gave up polling background response %s after %ss, so a later read through the proxy bills the key that "
        "created it",
        creator.response_id,
        schedule.timeout_seconds,
    )


async def _poll_logging_failures(
    creator: BackgroundResponseCreator,
    request: BackgroundResponsePollRequest,
    settle: SettlePolledResponse,
    fetch: FetchBackgroundResponse,
    schedule: PollSchedule,
) -> None:
    try:
        await poll_background_response(creator, request, settle, fetch, schedule)
    except Exception:  # noqa: BLE001  # a crashed poll keeps the creator row, so a later read still bills it
        verbose_proxy_logger.exception("Polling background response %s failed", creator.response_id)


_ACTIVE_POLLS: Final[set["asyncio.Task[None]"]] = set()  # mutable-ok: asyncio needs strong refs to poll tasks


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _proxy_database_configured() -> bool:
    from litellm.proxy.proxy_server import prisma_client  # noqa: PLC0415  # runtime global, set after load

    return prisma_client is not None


@dataclass(frozen=True, slots=True)
class BackgroundResponsePoller:
    fetch: FetchBackgroundResponse = HttpxBackgroundResponseFetch()
    schedule: PollSchedule = BACKGROUND_RESPONSE_POLL_SCHEDULE
    database_configured: Callable[[], bool] = _proxy_database_configured
    now: Callable[[], datetime] = _now

    def start(
        self,
        creator: BackgroundResponseCreator,
        request: BackgroundResponsePollRequest,
        settle: SettlePolledResponse,
    ) -> "asyncio.Task[None] | None":
        if not self.database_configured():
            return None
        task: Final = asyncio.create_task(_poll_logging_failures(creator, request, settle, self.fetch, self.schedule))
        _ACTIVE_POLLS.add(task)
        task.add_done_callback(_ACTIVE_POLLS.discard)
        return task


PollHeaders: TypeAlias = Callable[[str], Awaitable[Mapping[str, str] | None]]


def _origin(url: httpx.URL) -> tuple[str, str, int | None]:
    return (url.scheme, url.host, url.port)


def _resumable(creator: BackgroundResponseCreator, api_base_origin: tuple[str, str, int | None]) -> bool:
    return _origin(httpx.URL(creator.upstream_url)) == api_base_origin and is_pollable_response_id(creator.response_id)


async def resume_background_response_polls(
    creators: BackgroundResponseCreatorStore,
    poller: BackgroundResponsePoller,
    settle: SettlePolledResponse,
    api_base: str,
    poll_headers: PollHeaders,
) -> tuple["asyncio.Task[None]", ...]:
    try:
        return await _resume(creators, poller, settle, api_base, poll_headers)
    except Exception as e:  # noqa: BLE001  # a boot step must survive any failure; an unresumed row waits for a read
        verbose_proxy_logger.warning(
            "Could not resume background response polls, so a later read through the proxy bills the key that "
            "created each: %s",
            e,
        )
        return ()


async def _resume(
    creators: BackgroundResponseCreatorStore,
    poller: BackgroundResponsePoller,
    settle: SettlePolledResponse,
    api_base: str,
    poll_headers: PollHeaders,
) -> tuple["asyncio.Task[None]", ...]:
    api_base_origin: Final = _origin(httpx.URL(api_base))
    unsettled: Final = await creators.unsettled(poller.now() - timedelta(seconds=poller.schedule.timeout_seconds))
    resumable: Final = tuple(creator for creator in unsettled if _resumable(creator, api_base_origin))
    if not resumable:
        return ()
    headers: Final = await poll_headers(api_base)
    if headers is None:
        verbose_proxy_logger.warning(
            "Not resuming %s background response polls: the OpenAI pass-through has no credential, so a later "
            "read through the proxy bills the key that created each",
            len(resumable),
        )
        return ()
    started: Final = (
        poller.start(
            creator,
            BackgroundResponsePollRequest(
                url=response_url(httpx.URL(creator.upstream_url), creator.response_id),
                headers=tuple(headers.items()),
            ),
            settle,
        )
        for creator in resumable
    )
    return tuple(task for task in started if task is not None)
