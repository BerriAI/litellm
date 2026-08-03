"""
Types for passthrough managed-ID rewriting.

The Prisma client is an untyped runtime wrapper, so the managed-file /
managed-object rows and the table actions read by
``litellm.proxy.pass_through_endpoints.managed_id_rewriter`` are described
structurally here instead of being imported from generated stubs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol, TypedDict

PrismaWhere = dict[str, object]
PrismaOrder = dict[str, str] | list[dict[str, str]]


class ManagedResourceOwner(Protocol):
    """Ownership columns carried by every managed-resource row."""

    @property
    def created_by(self) -> str | None: ...

    @property
    def team_id(self) -> str | None: ...


class ManagedResourceRow(ManagedResourceOwner, Protocol):
    """A ``LiteLLM_ManagedFileTable`` or ``LiteLLM_ManagedObjectTable`` row.

    ``unified_file_id`` exists only on file rows and ``unified_object_id`` only
    on object rows; each is read exclusively off the table it belongs to.
    """

    @property
    def created_at(self) -> datetime | None: ...

    @property
    def file_object(self) -> object: ...

    @property
    def unified_file_id(self) -> str: ...

    @property
    def unified_object_id(self) -> str: ...


class ManagedResourceTable(Protocol):
    """The Prisma table actions the managed-ID rewriter issues."""

    async def find_first(self, *, where: PrismaWhere) -> ManagedResourceRow | None: ...

    async def find_many(
        self,
        *,
        where: PrismaWhere,
        order: PrismaOrder | None = None,
        take: int | None = None,
    ) -> Sequence[ManagedResourceRow]: ...

    async def update(self, *, where: PrismaWhere, data: PrismaWhere) -> ManagedResourceRow | None: ...

    async def upsert(self, *, where: PrismaWhere, data: Mapping[str, PrismaWhere]) -> ManagedResourceRow: ...


class PassthroughListResponse(TypedDict):
    """OpenAI-style paginated list body served from the managed-ID tables."""

    object: str
    data: Sequence[Mapping[str, object]]
    first_id: str | None
    last_id: str | None
    has_more: bool
