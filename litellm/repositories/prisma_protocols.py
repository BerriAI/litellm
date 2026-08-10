"""
Typed Protocol seams over prisma-client-py surfaces.

Modules that reach Prisma through an untyped handle (``prisma_client.db`` or a
repository ``.table``) annotate against these Protocols instead of hand-rolling
private ones per file.
"""

from collections.abc import Mapping, Sequence
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


class ReadOnlyTable(Protocol):
    async def find_many(self, *, where: Mapping[str, object]) -> Sequence[PrismaRecord]: ...


class SpendLinkedTable(Protocol[RowT_co]):
    async def find_many(self, *, where: Mapping[str, object]) -> Sequence[RowT_co]: ...

    async def update_many(self, *, where: Mapping[str, object], data: Mapping[str, object]) -> int: ...


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
