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
from litellm.constants import BACKGROUND_RESPONSE_SETTLEMENT_TIMEOUT_SECONDS
from litellm.interactions.background_cost_polling import is_carried_metadata_key, json_values, spend_metadata
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


class BackgroundResponseCreator(BaseModel):
    model_config = ConfigDict(frozen=True)

    response_id: str
    upstream_url: str
    context: BackgroundResponseCreateContext


def _carries(key: str) -> bool:
    return key not in BUDGET_RESERVATION_METADATA_KEYS and is_carried_metadata_key(key)


def upstream_url_without_query(url: httpx.URL) -> str:
    return str(httpx.URL(scheme=url.scheme, host=url.host, port=url.port, path=url.path.rstrip("/")))


def background_response_creator(
    logging_obj: LiteLLMLoggingObj,
    response_id: str,
    create_url: httpx.URL,
    model: str,
    custom_llm_provider: str | None,
    start_time: datetime,
) -> BackgroundResponseCreator:
    metadata: Final = spend_metadata(logging_obj)
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


BackgroundResponseSettlementDecision: TypeAlias = NotASettlement | SettledBefore | SettlesForCaller | SettlesForCreator

NOT_A_SETTLEMENT: Final = NotASettlement()
SETTLED_BEFORE: Final = SettledBefore()
SETTLES_FOR_CALLER: Final = SettlesForCaller()


def leaves_the_caller_unbilled(decision: BackgroundResponseSettlementDecision) -> bool:
    match decision:
        case SettledBefore() | SettlesForCreator():
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


class _CreatorKey(TypedDict):
    response_id: ReadOnly[str]


class _Later(TypedDict):
    gt: ReadOnly[datetime]


class _CreatedAfter(TypedDict):
    created_at: ReadOnly[_Later]


class BackgroundResponseCreatorTable(Protocol):
    def create(self, *, data: _NewCreatorRow) -> Awaitable[object]: ...

    def find_unique(self, *, where: _CreatorKey) -> Awaitable[_CreatorRow | None]: ...

    def find_many(self, *, where: _CreatedAfter) -> Awaitable[Sequence[_CreatorRow]]: ...

    def delete_many(self, *, where: _CreatorKey) -> Awaitable[int]: ...


class BackgroundResponseCreatorStore(Protocol):
    async def register(self, creator: BackgroundResponseCreator) -> None: ...

    async def creator(self, response_id: str) -> BackgroundResponseCreator | None: ...

    async def forget(self, response_id: str) -> None: ...

    async def unsettled(self, created_after: datetime) -> Sequence[BackgroundResponseCreator]: ...


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


_Result: Final = TypeVar("_Result")


@dataclass(frozen=True, slots=True)
class PrismaBackgroundResponseCreatorStore:
    table: Callable[[], BackgroundResponseCreatorTable | None] = _proxy_creator_table
    timeout_seconds: float = BACKGROUND_RESPONSE_SETTLEMENT_TIMEOUT_SECONDS

    async def register(self, creator: BackgroundResponseCreator) -> None:
        new_row: Final = _NewCreatorRow(
            response_id=creator.response_id,
            upstream_url=creator.upstream_url,
            create_context=_json(creator.context.model_dump(mode="json")),
        )
        await self._query(
            lambda table: table.create(data=new_row),
            None,
            f"record the creator of background response {creator.response_id}, so a read bills its reader",
        )

    async def creator(self, response_id: str) -> BackgroundResponseCreator | None:
        row: Final = await self._query(
            lambda table: table.find_unique(where=_CreatorKey(response_id=response_id)),
            None,
            f"look up the creator of background response {response_id}, so this read bills its reader",
        )
        return None if row is None else next(iter(_creators(row)), None)

    async def forget(self, response_id: str) -> None:
        await self._query(
            lambda table: table.delete_many(where=_CreatorKey(response_id=response_id)),
            0,
            f"forget the creator of settled background response {response_id}",
        )

    async def unsettled(self, created_after: datetime) -> Sequence[BackgroundResponseCreator]:
        rows: Final = await self._query(
            lambda table: table.find_many(where=_CreatedAfter(created_at=_Later(gt=created_after))),
            (),
            "list the unsettled background responses to resume polling",
        )
        return tuple(chain.from_iterable(_creators(row) for row in rows))

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
