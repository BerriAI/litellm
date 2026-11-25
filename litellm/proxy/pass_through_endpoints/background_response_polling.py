import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Final, TypeAlias
from urllib.parse import quote

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.integrations.otel.runtime import background_job_span
from litellm.interactions.background_cost_polling import PollSchedule, poll_intervals
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared client factory has an untyped params dict
)
from litellm.types.llms.custom_http import httpxSpecialProvider

from .background_response_creator import (
    BackgroundResponseCreator,
    BackgroundResponseCreatorStore,
    CreatorGone,
    LeaseRenewed,
    LeaseUnrecorded,
)

BACKGROUND_RESPONSE_POLL_SCHEDULE: Final = PollSchedule(
    initial_interval_seconds=5.0, max_interval_seconds=60.0, timeout_seconds=7800.0
)

BACKGROUND_RESPONSE_POLL_LEASE_SECONDS: Final = 300.0
BACKGROUND_RESPONSE_MAX_ACTIVE_POLLS: Final = 1000
BACKGROUND_RESPONSE_FETCH_TIMEOUT_SECONDS: Final = 10.0

BACKGROUND_RESPONSE_SWEEP_INTERVAL_SECONDS: Final = 120.0

BACKGROUND_RESPONSE_CREATOR_RETENTION: Final = timedelta(days=30)

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


PollHeaders: TypeAlias = Callable[[str], Awaitable[Mapping[str, str] | None]]


async def _openai_passthrough_poll_headers(api_base: str) -> Mapping[str, str] | None:
    from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (  # noqa: PLC0415  # that module imports this one
        openai_passthrough_poll_headers,
    )

    return await openai_passthrough_poll_headers(api_base)


def _api_base(url: httpx.URL) -> str:
    return str(url.copy_with(path="/", query=None, fragment=None))


def _authorization(headers: Iterable[tuple[str, str]]) -> str | None:
    return next((value for name, value in headers if name.lower() == "authorization"), None)


def _with_credential(
    headers: tuple[tuple[str, str], ...], credential: Mapping[str, str]
) -> tuple[tuple[str, str], ...]:
    replaced: Final = frozenset(name.lower() for name in credential)
    kept: Final = tuple((name, value) for name, value in headers if name.lower() not in replaced)
    return kept + tuple(credential.items())


@dataclass(frozen=True, slots=True)
class HttpxBackgroundResponseFetch:
    client: Callable[[], httpx.AsyncClient] = _pass_through_client
    timeout_seconds: float = BACKGROUND_RESPONSE_FETCH_TIMEOUT_SECONDS
    current_credential: PollHeaders = _openai_passthrough_poll_headers

    async def __call__(self, request: BackgroundResponsePollRequest) -> FetchOutcome:
        match await self._get(request.url, request.headers):
            case StopPolling(status_code=401) as unauthorized:
                refreshed: Final = await self._refreshed_headers(request)
                return unauthorized if refreshed is None else await self._get(request.url, refreshed)
            case outcome:
                return outcome

    async def _get(self, url: httpx.URL, headers: tuple[tuple[str, str], ...]) -> FetchOutcome:
        try:
            response: Final = await self.client().get(url, headers=list(headers), timeout=self.timeout_seconds)
        except httpx.HTTPError as e:
            return RetryLater(reason=f"{type(e).__name__}: {e}")
        return _fetch_outcome(response)

    async def _refreshed_headers(self, request: BackgroundResponsePollRequest) -> tuple[tuple[str, str], ...] | None:
        try:
            credential: Final = await self.current_credential(_api_base(request.url))
        except Exception as e:  # noqa: BLE001  # a failed token exchange stops the poll as the 401 alone would
            verbose_proxy_logger.debug("Could not refresh the credential for background response polls: %s", e)
            return None
        if credential is None or _authorization(credential.items()) == _authorization(request.headers):
            return None
        return _with_credential(request.headers, credential)


@dataclass(frozen=True, slots=True)
class PollLease:
    creators: BackgroundResponseCreatorStore
    now: Callable[[], datetime]
    lease_seconds: float
    hold_seconds: float

    async def renew(self, response_id: str) -> bool:
        match await self.creators.renew_lease(response_id, self.now() + timedelta(seconds=self.lease_seconds)):
            case CreatorGone():
                return False
            case LeaseRenewed() | LeaseUnrecorded():
                return True

    async def hold(self, response_id: str) -> None:
        await self.creators.renew_lease(response_id, self.now() + timedelta(seconds=self.hold_seconds))


async def poll_background_response(
    creator: BackgroundResponseCreator,
    request: BackgroundResponsePollRequest,
    settle: SettlePolledResponse,
    fetch: FetchBackgroundResponse,
    schedule: PollSchedule,
    lease: PollLease,
) -> None:
    intervals: Final = poll_intervals(
        schedule.initial_interval_seconds, schedule.max_interval_seconds, schedule.timeout_seconds
    )
    for interval in intervals:
        await asyncio.sleep(interval)
        if not await lease.renew(creator.response_id):
            verbose_proxy_logger.debug(
                "Stopped polling background response %s: another settler already billed it", creator.response_id
            )
            return
        match await fetch(request):
            case Fetched(body=body) if body.get("status") in POLLABLE_BACKGROUND_RESPONSE_STATUSES:
                continue
            case Fetched(body=body):
                if await settle(creator, body):
                    return
                verbose_proxy_logger.debug("Could not settle background response %s, will retry", creator.response_id)
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
                await lease.hold(creator.response_id)
                return
    verbose_proxy_logger.warning(
        "Gave up polling background response %s after %ss, so a later read through the proxy bills the key that "
        "created it",
        creator.response_id,
        schedule.timeout_seconds,
    )
    await lease.hold(creator.response_id)


async def _poll_logging_failures(
    creator: BackgroundResponseCreator,
    request: BackgroundResponsePollRequest,
    settle: SettlePolledResponse,
    fetch: FetchBackgroundResponse,
    schedule: PollSchedule,
    lease: PollLease,
) -> None:
    try:
        await poll_background_response(creator, request, settle, fetch, schedule, lease)
    except Exception:  # noqa: BLE001  # a crashed poll keeps the creator row, so a sweep or a later read still bills it
        verbose_proxy_logger.exception("Polling background response %s failed", creator.response_id)


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
    lease_seconds: float = BACKGROUND_RESPONSE_POLL_LEASE_SECONDS
    max_active_polls: int = BACKGROUND_RESPONSE_MAX_ACTIVE_POLLS
    active: dict[str, "asyncio.Task[None]"] = field(  # mutable-ok: asyncio needs strong refs to the polls it runs
        default_factory=dict
    )

    def lease_until(self) -> datetime:
        return self.now() + timedelta(seconds=self.lease_seconds)

    def polls(self, response_id: str) -> bool:
        return response_id in self.active

    def free_poll_slots(self) -> int:
        return max(self.max_active_polls - len(self.active), 0)

    def start(
        self,
        creator: BackgroundResponseCreator,
        request: BackgroundResponsePollRequest,
        settle: SettlePolledResponse,
        creators: BackgroundResponseCreatorStore,
    ) -> "asyncio.Task[None] | None":
        if not self.database_configured() or self.polls(creator.response_id):
            return None
        if self.free_poll_slots() == 0:
            verbose_proxy_logger.warning(
                "Not polling background response %s yet: this worker already polls %s, so a later sweep polls it",
                creator.response_id,
                self.max_active_polls,
            )
            return None
        lease: Final = PollLease(
            creators=creators,
            now=self.now,
            lease_seconds=self.lease_seconds,
            hold_seconds=self.schedule.timeout_seconds,
        )
        task: Final = asyncio.create_task(
            _poll_logging_failures(creator, request, settle, self.fetch, self.schedule, lease)
        )
        self.active[creator.response_id] = task
        task.add_done_callback(lambda _: self.active.pop(creator.response_id, None))
        return task


def _origin(url: httpx.URL) -> tuple[str, str, int | None]:
    return (url.scheme, url.host, url.port)


def _resumable(creator: BackgroundResponseCreator, api_base_origin: tuple[str, str, int | None]) -> bool:
    return _origin(httpx.URL(creator.upstream_url)) == api_base_origin and is_pollable_response_id(creator.response_id)


def _resumed_poll_request(
    creator: BackgroundResponseCreator, headers: Mapping[str, str]
) -> BackgroundResponsePollRequest:
    return BackgroundResponsePollRequest(
        url=response_url(httpx.URL(creator.upstream_url), creator.response_id), headers=tuple(headers.items())
    )


def pollable_at(creator: BackgroundResponseCreator, url: httpx.URL) -> bool:
    return _resumable(creator, _origin(url))


async def sweep_background_response_polls(
    creators: BackgroundResponseCreatorStore,
    poller: BackgroundResponsePoller,
    settle: SettlePolledResponse,
    api_base: str,
    poll_headers: PollHeaders,
) -> tuple["asyncio.Task[None]", ...]:
    try:
        return await _sweep(creators, poller, settle, api_base, poll_headers)
    except Exception as e:  # noqa: BLE001  # a failed sweep must not end the sweeps; an unpolled row waits for the next
        verbose_proxy_logger.warning(
            "Could not resume background response polls, so the next sweep or a later read through the proxy bills "
            "the key that created each: %s",
            e,
        )
        return ()


async def _sweep(
    creators: BackgroundResponseCreatorStore,
    poller: BackgroundResponsePoller,
    settle: SettlePolledResponse,
    api_base: str,
    poll_headers: PollHeaders,
) -> tuple["asyncio.Task[None]", ...]:
    with background_job_span("sweep_background_response_polls"):
        now: Final = poller.now()
        await creators.prune(now - BACKGROUND_RESPONSE_CREATOR_RETENTION)
        api_base_origin: Final = _origin(httpx.URL(api_base))
        unleased: Final = await creators.unleased(now - timedelta(seconds=poller.schedule.timeout_seconds), now)
        resumable: Final = tuple(
            creator
            for creator in unleased
            if _resumable(creator, api_base_origin) and not poller.polls(creator.response_id)
        )[: poller.free_poll_slots()]
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
        taken: Final = await asyncio.gather(
            *(creators.take_lease(creator.response_id, now, poller.lease_until()) for creator in resumable)
        )
    leased: Final = tuple(creator for creator, took in zip(resumable, taken, strict=True) if took)
    started: Final = (
        poller.start(creator, _resumed_poll_request(creator, headers), settle, creators) for creator in leased
    )
    return tuple(task for task in started if task is not None)


async def sweep_periodically(sweep: Callable[[], Awaitable[object]], interval_seconds: float) -> None:
    while True:
        await sweep()
        await asyncio.sleep(interval_seconds)
