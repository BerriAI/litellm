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


class PrismaRecord(Protocol):
    def dict(self) -> Mapping[str, object]: ...


class ReadOnlyTable(Protocol):
    async def find_many(self, *, where: Mapping[str, object]) -> Sequence[PrismaRecord]: ...


class SpendLinkedTable(Protocol[RowT_co]):
    async def find_many(self, *, where: Mapping[str, object]) -> Sequence[RowT_co]: ...

    async def update_many(self, *, where: Mapping[str, object], data: Mapping[str, object]) -> int: ...


class TableActions(Protocol[RowT]):
    """Structural view of the generated Prisma actions for a single table."""

    async def find_many(
        self,
        *,
        where: Mapping[str, object] | None = None,
        take: int | None = None,
        skip: int | None = None,
        order: Mapping[str, str] | Sequence[Mapping[str, str]] | None = None,
        include: Mapping[str, object] | None = None,
    ) -> list[RowT]: ...

    async def find_first(
        self,
        *,
        where: Mapping[str, object] | None = None,
        skip: int | None = None,
        order: Mapping[str, str] | Sequence[Mapping[str, str]] | None = None,
        include: Mapping[str, object] | None = None,
    ) -> RowT | None: ...

    async def find_unique(
        self,
        *,
        where: Mapping[str, object],
        include: Mapping[str, object] | None = None,
    ) -> RowT | None: ...

    async def create(
        self,
        data: Mapping[str, object],
        include: Mapping[str, object] | None = None,
    ) -> RowT: ...

    async def create_many(
        self,
        data: Sequence[Mapping[str, object]],
        *,
        skip_duplicates: bool | None = None,
    ) -> int: ...

    async def update(
        self,
        data: Mapping[str, object],
        where: Mapping[str, object],
        include: Mapping[str, object] | None = None,
    ) -> RowT | None: ...

    async def update_many(self, *, where: Mapping[str, object], data: Mapping[str, object]) -> int: ...

    async def upsert(
        self,
        *,
        where: Mapping[str, object],
        data: Mapping[str, Mapping[str, object]],
        include: Mapping[str, object] | None = None,
    ) -> RowT: ...

    async def delete(
        self,
        *,
        where: Mapping[str, object],
        include: Mapping[str, object] | None = None,
    ) -> RowT | None: ...

    async def delete_many(self, *, where: Mapping[str, object] | None = None) -> int: ...

    async def count(self, *, where: Mapping[str, object] | None = None) -> int: ...


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
