"""
Typed surfaces for the passthrough managed-ID rewriter.

Prisma's generated client is untyped at the ``litellm`` boundary, so the row
shapes, table actions, and query fragments the rewriter touches are declared
here as protocols instead of leaking ``Any`` through every call site.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Literal,
    Protocol,
    TypeAlias,
    TypedDict,
    TypeVar,
    runtime_checkable,
)

from pydantic import JsonValue

if TYPE_CHECKING:
    from litellm.models.managed_files import LiteLLM_ManagedFileTable
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.llms.openai import OpenAIFileObject

SortOrder: TypeAlias = Literal["asc", "desc"]
ResourceKind: TypeAlias = Literal["files", "batches"]

PrismaWhereValue: TypeAlias = (
    "str | int | bool | datetime | None | Mapping[str, PrismaWhereValue] | Sequence[PrismaWhereValue]"
)
PrismaWhere: TypeAlias = "Mapping[str, PrismaWhereValue]"
PrismaOrder: TypeAlias = "Mapping[str, SortOrder]"
ManagedRowData: TypeAlias = "Mapping[str, str | None]"


class ManagedResourceRow(Protocol):
    """Columns shared by ``LiteLLM_ManagedFileTable`` and ``LiteLLM_ManagedObjectTable`` rows."""

    created_by: str | None
    team_id: str | None
    created_at: datetime | None
    file_object: JsonValue


class ManagedFileRow(ManagedResourceRow, Protocol):
    unified_file_id: str


class ManagedObjectRow(ManagedResourceRow, Protocol):
    unified_object_id: str


RowT = TypeVar(
    "RowT", bound=ManagedResourceRow
)  # rebind-ok: TypeVar declarations must stay bare assignments for pyright


class ManagedTable(Protocol[RowT]):
    """The Prisma table actions the rewriter reads rows through."""

    async def find_first(self, *, where: PrismaWhere) -> RowT | None: ...

    async def find_many(
        self,
        *,
        where: PrismaWhere,
        order: PrismaOrder | Sequence[PrismaOrder] | None = None,
        take: int | None = None,
    ) -> list[RowT]: ...


class ManagedFileTable(ManagedTable[ManagedFileRow], Protocol): ...


class ManagedObjectTable(ManagedTable[ManagedObjectRow], Protocol):
    async def update(self, *, where: PrismaWhere, data: ManagedRowData) -> ManagedObjectRow | None: ...

    async def upsert(self, *, where: PrismaWhere, data: Mapping[str, ManagedRowData]) -> ManagedObjectRow: ...


@runtime_checkable
class ManagedFileIdReader(Protocol):
    """Row lookup on the enterprise managed-files hook.

    The proxy hook registry is untyped and hands back a bare ``CustomLogger``,
    so this protocol is an ``isinstance`` target: the rewriter checks the method
    is really there before calling it.  It is kept separate from
    ``ManagedFileIdWriter`` so a hook implementing only one of the two is
    narrowed on exactly the capability about to be used.
    """

    async def get_unified_file_id(
        self,
        file_id: str,
        litellm_parent_otel_span: object = None,
    ) -> LiteLLM_ManagedFileTable | None: ...


@runtime_checkable
class ManagedFileIdWriter(Protocol):
    """Row persistence on the enterprise managed-files hook."""

    async def store_unified_file_id(
        self,
        file_id: str,
        file_object: OpenAIFileObject | None,
        litellm_parent_otel_span: object,
        model_mappings: dict[str, str],
        user_api_key_dict: UserAPIKeyAuth,
    ) -> None: ...


class ManagedListResponse(TypedDict):
    """OpenAI-style paginated list body served from the managed-resource tables."""

    object: Literal["list"]
    data: list[dict[str, JsonValue]]
    first_id: str | None
    last_id: str | None
    has_more: bool
