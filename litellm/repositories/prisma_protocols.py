"""
Typed Protocol seams over prisma-client-py surfaces.

Modules that reach Prisma through an untyped handle (``prisma_client.db`` or a
repository ``.table``) annotate against these Protocols instead of hand-rolling
private ones per file.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol, TypeVar

RowT_co = TypeVar("RowT_co", covariant=True)
RowT = TypeVar("RowT")


PrismaOrderBy = Mapping[str, object] | Sequence[Mapping[str, object]]


class TableActions(Protocol[RowT]):
    """The subset of prisma-client-py table actions reached through an untyped handle."""

    async def find_unique(
        self,
        *,
        where: Mapping[str, object],
        include: Mapping[str, object] | None = None,
    ) -> RowT | None: ...

    async def find_first(
        self,
        *,
        where: Mapping[str, object],
        include: Mapping[str, object] | None = None,
        order: PrismaOrderBy | None = None,
    ) -> RowT | None: ...

    async def find_many(
        self,
        *,
        where: Mapping[str, object] | None = None,
        include: Mapping[str, object] | None = None,
        order: PrismaOrderBy | None = None,
        skip: int | None = None,
        take: int | None = None,
    ) -> list[RowT]: ...

    async def count(self, *, where: Mapping[str, object] | None = None) -> int: ...

    async def create(
        self,
        *,
        data: Mapping[str, object],
        include: Mapping[str, object] | None = None,
    ) -> RowT: ...

    async def update(
        self,
        *,
        where: Mapping[str, object],
        data: Mapping[str, object],
        include: Mapping[str, object] | None = None,
    ) -> RowT | None: ...

    async def update_many(self, *, where: Mapping[str, object], data: Mapping[str, object]) -> int: ...

    async def delete_many(self, *, where: Mapping[str, object] | None = None) -> int: ...


class PrismaRecord(Protocol):
    def dict(self) -> Mapping[str, object]: ...


class PrismaTableActions(Protocol):
    """The subset of prisma-client-py table actions the repositories rely on."""

    async def find_unique(
        self, *, where: Mapping[str, object], include: Mapping[str, object] | None = None
    ) -> PrismaRecord | None: ...

    async def find_first(
        self, *, where: Mapping[str, object], include: Mapping[str, object] | None = None
    ) -> PrismaRecord | None: ...

    async def find_many(
        self,
        *,
        where: Mapping[str, object] | None = None,
        include: Mapping[str, object] | None = None,
        skip: int | None = None,
        take: int | None = None,
        order: Mapping[str, str] | None = None,
    ) -> Sequence[PrismaRecord]: ...

    async def create(
        self, *, data: Mapping[str, object], include: Mapping[str, object] | None = None
    ) -> PrismaRecord: ...

    async def update(
        self, *, where: Mapping[str, object], data: Mapping[str, object]
    ) -> PrismaRecord | None: ...

    async def delete(self, *, where: Mapping[str, object]) -> PrismaRecord | None: ...

    async def count(self, *, where: Mapping[str, object] | None = None) -> int: ...


class ReadOnlyTable(Protocol):
    async def find_many(self, *, where: Mapping[str, object]) -> Sequence[PrismaRecord]: ...


class SpendLinkedTable(Protocol[RowT_co]):
    async def find_many(self, *, where: Mapping[str, object]) -> Sequence[RowT_co]: ...

    async def update_many(self, *, where: Mapping[str, object], data: Mapping[str, object]) -> int: ...


class MemoryRecord(Protocol):
    memory_id: str
    key: str
    value: str
    metadata: object | None
    user_id: str | None
    team_id: str | None
    created_at: datetime | None
    created_by: str | None
    updated_at: datetime | None
    updated_by: str | None


class MemoryTable(Protocol):
    async def create(self, *, data: Mapping[str, object]) -> MemoryRecord: ...

    async def count(self, *, where: Mapping[str, object]) -> int: ...

    async def find_many(
        self,
        *,
        where: Mapping[str, object],
        order: Mapping[str, str],
        skip: int = 0,
        take: int | None = None,
    ) -> Sequence[MemoryRecord]: ...

    async def update(self, *, where: Mapping[str, object], data: Mapping[str, object]) -> MemoryRecord: ...

    async def delete(self, *, where: Mapping[str, object]) -> MemoryRecord | None: ...


class ToolIndexRecord(Protocol):
    request_id: str


class ToolIndexTable(Protocol):
    async def count(self, *, where: Mapping[str, object]) -> int: ...

    async def find_many(
        self,
        *,
        where: Mapping[str, object],
        order: Mapping[str, str],
        skip: int = 0,
        take: int | None = None,
    ) -> Sequence[ToolIndexRecord]: ...


class SpendLogUsageRecord(Protocol):
    request_id: str
    startTime: datetime
    model: str | None
    spend: float | None
    total_tokens: int | None
    messages: object | None
    proxy_server_request: object | None


class SpendLogUsageTable(Protocol):
    async def find_many(self, *, where: Mapping[str, object]) -> Sequence[SpendLogUsageRecord]: ...


class DailyToolSpendRecord(Protocol):
    date: str
    tool_name: str
    spend: float
    request_count: int


class DailyToolSpendTable(Protocol):
    async def group_by(
        self,
        *,
        by: Sequence[str],
        sum: Mapping[str, bool],
        where: Mapping[str, object],
        order: Mapping[str, object],
        take: int,
    ) -> Sequence[Mapping[str, object]] | None: ...

    async def find_many(
        self,
        *,
        where: Mapping[str, object],
        order: Sequence[Mapping[str, str]],
    ) -> Sequence[DailyToolSpendRecord]: ...


class ObjectPermissionOwnerRecord(Protocol):
    object_permission_id: str | None


class ObjectPermissionOwnerTable(Protocol):
    async def find_unique(self, *, where: Mapping[str, object]) -> ObjectPermissionOwnerRecord | None: ...

    async def update_many(self, *, where: Mapping[str, object], data: Mapping[str, object]) -> int: ...


class ObjectPermissionWriteTable(Protocol):
    async def create(self, *, data: Mapping[str, object]) -> object: ...

    async def delete(self, *, where: Mapping[str, object]) -> object: ...


class BatchTable(Protocol):
    def update(self, *, where: Mapping[str, object], data: Mapping[str, object]) -> None: ...


class PrismaBatch(Protocol):
    @property
    def litellm_verificationtoken(self) -> BatchTable: ...

    @property
    def litellm_usertable(self) -> BatchTable: ...

    @property
    def litellm_teamtable(self) -> BatchTable: ...

    async def commit(self) -> None: ...
