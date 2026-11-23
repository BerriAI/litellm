import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import chain
from typing import Final, Protocol, TypeAlias, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.constants import BACKGROUND_RESPONSE_SETTLEMENT_TIMEOUT_SECONDS, X_LITELLM_DISABLE_CALLBACKS
from litellm.interactions.background_cost_polling import is_carried_metadata_key, json_values, spend_metadata
from litellm.litellm_core_utils.core_helpers import as_str_mapping, proxy_request_headers
from litellm.litellm_core_utils.internal_call_metadata import BUDGET_RESERVATION_METADATA_KEYS
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.repositories.table_repositories import BackgroundResponseCreatorRepository


class BackgroundResponseCreateContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    custom_llm_provider: str | None
    litellm_call_id: str
    litellm_trace_id: str
    start_time: datetime
    metadata: Mapping[str, JsonValue]
    no_log: bool = False
    disabled_callbacks: tuple[str, ...] = ()
    key_or_team_logging: bool = False


class BackgroundResponseCreator(BaseModel):
    model_config = ConfigDict(frozen=True)

    response_id: str
    upstream_url: str
    context: BackgroundResponseCreateContext


def _carries(key: str) -> bool:
    return key not in BUDGET_RESERVATION_METADATA_KEYS and is_carried_metadata_key(key)


def upstream_url_without_query(url: httpx.URL) -> str:
    return str(httpx.URL(scheme=url.scheme, host=url.host, port=url.port, path=url.path.rstrip("/")))


def _disabled_callbacks(
    litellm_params: Mapping[str, object] | None, requested: Sequence[str] | None
) -> tuple[str, ...]:
    header: Final = (proxy_request_headers(litellm_params) or {}).get(X_LITELLM_DISABLE_CALLBACKS)
    if isinstance(header, str):
        return tuple(sorted({name.strip().lower() for name in header.split(",")}))
    return tuple(requested or ())


def _has_key_or_team_logging(logging_obj: LiteLLMLoggingObj) -> bool:
    dynamic_callbacks: Final = (
        logging_obj.dynamic_success_callbacks,
        logging_obj.dynamic_async_success_callbacks,
        logging_obj.dynamic_failure_callbacks,
        logging_obj.dynamic_async_failure_callbacks,
    )
    return any(dynamic_callbacks) or any(
        value is not None
        for name, value in logging_obj.standard_callback_dynamic_params.items()
        if name != "litellm_disabled_callbacks"
    )


def background_response_creator(
    logging_obj: LiteLLMLoggingObj,
    response_id: str,
    create_url: httpx.URL,
    model: str,
    custom_llm_provider: str | None,
    start_time: datetime,
) -> BackgroundResponseCreator:
    metadata: Final = spend_metadata(logging_obj)
    litellm_params: Final = as_str_mapping(logging_obj.model_call_details.get("litellm_params"))
    return BackgroundResponseCreator(
        response_id=response_id,
        upstream_url=upstream_url_without_query(create_url),
        context=BackgroundResponseCreateContext(
            model=model,
            custom_llm_provider=custom_llm_provider,
            litellm_call_id=logging_obj.litellm_call_id,
            litellm_trace_id=logging_obj.litellm_trace_id,
            start_time=start_time,
            metadata=json_values((key, value) for key, value in metadata.items() if _carries(key)),
            no_log=(litellm_params or {}).get("no-log") is True,
            disabled_callbacks=_disabled_callbacks(
                litellm_params, logging_obj.standard_callback_dynamic_params.get("litellm_disabled_callbacks")
            ),
            key_or_team_logging=_has_key_or_team_logging(logging_obj),
        ),
    )


@dataclass(frozen=True, slots=True)
class NotASettlement:
    pass


@dataclass(frozen=True, slots=True)
class SettledBefore:
    pass


@dataclass(frozen=True, slots=True)
class SettlesForCaller:
    pass


@dataclass(frozen=True, slots=True)
class SettlesForCreator:
    creator: BackgroundResponseCreator
    response_body: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class CreatorBillDeferred:
    pass


@dataclass(frozen=True, slots=True)
class CreatorLookupFailed:
    pass


CREATOR_LOOKUP_FAILED: Final = CreatorLookupFailed()


BackgroundResponseSettlementDecision: TypeAlias = (
    NotASettlement | SettledBefore | SettlesForCaller | SettlesForCreator | CreatorBillDeferred
)

NOT_A_SETTLEMENT: Final = NotASettlement()
SETTLED_BEFORE: Final = SettledBefore()
SETTLES_FOR_CALLER: Final = SettlesForCaller()
CREATOR_BILL_DEFERRED: Final = CreatorBillDeferred()


def leaves_the_caller_unbilled(decision: BackgroundResponseSettlementDecision) -> bool:
    match decision:
        case SettledBefore() | SettlesForCreator() | CreatorBillDeferred():
            return True
        case NotASettlement() | SettlesForCaller():
            return False


class _CreatorRow(Protocol):
    @property
    def response_id(self) -> str: ...
    @property
    def upstream_url(self) -> str: ...
    @property
    def create_context(self) -> object: ...


class _NewCreatorRow(TypedDict):
    response_id: ReadOnly[str]
    upstream_url: ReadOnly[str]
    create_context: ReadOnly[object]
    poll_lease_expires_at: ReadOnly[datetime]


class _CreatorKey(TypedDict):
    response_id: ReadOnly[str]


class _Later(TypedDict):
    gt: ReadOnly[datetime]


class _Earlier(TypedDict):
    lt: ReadOnly[datetime]


class _LeaseExpiry(TypedDict):
    poll_lease_expires_at: ReadOnly[_Earlier | None]


class _UnleasedCreator(TypedDict):
    response_id: ReadOnly[str]
    OR: ReadOnly[tuple[_LeaseExpiry, ...]]


class _UnleasedSince(TypedDict):
    created_at: ReadOnly[_Later]
    OR: ReadOnly[tuple[_LeaseExpiry, ...]]


class _CreatedBefore(TypedDict):
    created_at: ReadOnly[_Earlier]


class _Lease(TypedDict):
    poll_lease_expires_at: ReadOnly[datetime]


class BackgroundResponseCreatorTable(Protocol):
    def create(self, *, data: _NewCreatorRow) -> Awaitable[object]: ...

    def find_unique(self, *, where: _CreatorKey) -> Awaitable[_CreatorRow | None]: ...

    def find_many(self, *, where: _UnleasedSince) -> Awaitable[Sequence[_CreatorRow]]: ...

    def update_many(self, *, data: _Lease, where: _CreatorKey | _UnleasedCreator) -> Awaitable[int]: ...

    def delete_many(self, *, where: _CreatorKey | _CreatedBefore) -> Awaitable[int]: ...


@dataclass(frozen=True, slots=True)
class LeaseRenewed:
    pass


@dataclass(frozen=True, slots=True)
class CreatorGone:
    pass


@dataclass(frozen=True, slots=True)
class LeaseUnrecorded:
    pass


LeaseRenewal: TypeAlias = LeaseRenewed | CreatorGone | LeaseUnrecorded

LEASE_RENEWED: Final = LeaseRenewed()
CREATOR_GONE: Final = CreatorGone()
LEASE_UNRECORDED: Final = LeaseUnrecorded()


class BackgroundResponseCreatorStore(Protocol):
    async def register(self, creator: BackgroundResponseCreator, poll_lease_expires_at: datetime) -> None: ...

    async def creator(self, response_id: str) -> BackgroundResponseCreator | CreatorLookupFailed | None: ...

    async def forget(self, response_id: str) -> None: ...

    async def unleased(self, created_after: datetime, now: datetime) -> Sequence[BackgroundResponseCreator]: ...

    async def take_lease(self, response_id: str, now: datetime, until: datetime) -> bool: ...

    async def renew_lease(self, response_id: str, until: datetime) -> LeaseRenewal: ...

    async def prune(self, created_before: datetime) -> None: ...


def _proxy_creator_table() -> BackgroundResponseCreatorTable | None:
    from litellm.proxy.proxy_server import prisma_client  # noqa: PLC0415  # runtime global, set after load

    if prisma_client is None:
        return None
    return BackgroundResponseCreatorRepository(prisma_client).table


def _json(data: Mapping[str, object]) -> object:
    from prisma import Json  # noqa: PLC0415  # local import: prisma may be ungenerated at module load in some tools

    return Json(dict(data))


def _creators(row: _CreatorRow) -> tuple[BackgroundResponseCreator, ...]:
    try:
        context: Final = BackgroundResponseCreateContext.model_validate(row.create_context)
    except ValidationError:
        verbose_proxy_logger.warning(
            "Background response %s has a creator row this version cannot read; a read bills its reader",
            row.response_id,
        )
        return ()
    return (BackgroundResponseCreator(response_id=row.response_id, upstream_url=row.upstream_url, context=context),)


def _lease_expired_by(now: datetime) -> tuple[_LeaseExpiry, ...]:
    return (_LeaseExpiry(poll_lease_expires_at=None), _LeaseExpiry(poll_lease_expires_at=_Earlier(lt=now)))


_Result = TypeVar("_Result")


@dataclass(frozen=True, slots=True)
class PrismaBackgroundResponseCreatorStore:
    table: Callable[[], BackgroundResponseCreatorTable | None] = _proxy_creator_table
    timeout_seconds: float = BACKGROUND_RESPONSE_SETTLEMENT_TIMEOUT_SECONDS

    async def register(self, creator: BackgroundResponseCreator, poll_lease_expires_at: datetime) -> None:
        new_row: Final = _NewCreatorRow(
            response_id=creator.response_id,
            upstream_url=creator.upstream_url,
            create_context=_json(creator.context.model_dump(mode="json")),
            poll_lease_expires_at=poll_lease_expires_at,
        )
        await self._query(
            lambda table: table.create(data=new_row),
            None,
            f"record the creator of background response {creator.response_id}, so a read bills its reader",
        )

    async def creator(self, response_id: str) -> BackgroundResponseCreator | CreatorLookupFailed | None:
        from prisma.errors import TableNotFoundError  # noqa: PLC0415  # local import: prisma may be ungenerated at load

        table: Final = self.table()
        if table is None:
            return None
        try:
            row: Final = await asyncio.wait_for(
                table.find_unique(where=_CreatorKey(response_id=response_id)), timeout=self.timeout_seconds
            )
        except TableNotFoundError:
            return None
        except Exception as e:  # noqa: BLE001  # an unknown creator defers the bill instead of billing the reader
            verbose_proxy_logger.warning("Could not look up the creator of background response %s: %s", response_id, e)
            return CREATOR_LOOKUP_FAILED
        return None if row is None else next(iter(_creators(row)), None)

    async def forget(self, response_id: str) -> None:
        await self._query(
            lambda table: table.delete_many(where=_CreatorKey(response_id=response_id)),
            0,
            f"forget the creator of settled background response {response_id}",
        )

    async def unleased(self, created_after: datetime, now: datetime) -> Sequence[BackgroundResponseCreator]:
        rows: Final = await self._query(
            lambda table: table.find_many(
                where=_UnleasedSince(created_at=_Later(gt=created_after), OR=_lease_expired_by(now))
            ),
            (),
            "list the unleased background responses to resume polling",
        )
        return tuple(chain.from_iterable(_creators(row) for row in rows))

    async def take_lease(self, response_id: str, now: datetime, until: datetime) -> bool:
        taken: Final = await self._query(
            lambda table: table.update_many(
                data=_Lease(poll_lease_expires_at=until),
                where=_UnleasedCreator(response_id=response_id, OR=_lease_expired_by(now)),
            ),
            0,
            f"take the poll lease of background response {response_id}",
        )
        return taken > 0

    async def renew_lease(self, response_id: str, until: datetime) -> LeaseRenewal:
        renewed: Final = await self._query(
            lambda table: table.update_many(
                data=_Lease(poll_lease_expires_at=until), where=_CreatorKey(response_id=response_id)
            ),
            None,
            f"renew the poll lease of background response {response_id}",
        )
        if renewed is None:
            return LEASE_UNRECORDED
        return LEASE_RENEWED if renewed > 0 else CREATOR_GONE

    async def prune(self, created_before: datetime) -> None:
        await self._query(
            lambda table: table.delete_many(where=_CreatedBefore(created_at=_Earlier(lt=created_before))),
            0,
            "delete the creators of background responses past their retention",
        )

    async def _query(
        self,
        query: Callable[[BackgroundResponseCreatorTable], Awaitable[_Result]],
        when_unavailable: _Result,
        action: str,
    ) -> _Result:
        try:
            table: Final = self.table()
            if table is None:
                return when_unavailable
            return await asyncio.wait_for(query(table), timeout=self.timeout_seconds)
        except Exception as e:  # noqa: BLE001  # a creator store failure degrades to billing the reader as before
            verbose_proxy_logger.warning("Could not %s: %s", action, e)
            return when_unavailable
