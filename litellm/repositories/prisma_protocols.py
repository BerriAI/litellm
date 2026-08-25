"""
Typed Protocol seams over prisma-client-py surfaces.

Modules that reach Prisma through an untyped handle (``prisma_client.db`` or a
repository ``.table``) annotate against these Protocols instead of hand-rolling
private ones per file.
"""

from collections.abc import Mapping, Sequence
from typing import Protocol, TypeVar

RowT_co = TypeVar("RowT_co", covariant=True)


class TableActions(Protocol[RowT_co]):
    """The prisma-client-py per-model action surface, keyed to the row it returns.

    Query inputs stay `Mapping[str, object]` rather than the generated
    `types.*` TypedDicts so callers can keep passing plain dicts, while every
    result carries the row type the repository is bound to.
    """

    async def find_unique(
        self, where: Mapping[str, object], include: Mapping[str, object] | None = None
    ) -> RowT_co | None: ...

    async def find_first(
        self,
        skip: int | None = None,
        where: Mapping[str, object] | None = None,
        cursor: Mapping[str, object] | None = None,
        include: Mapping[str, object] | None = None,
        order: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
        distinct: Sequence[str] | None = None,
    ) -> RowT_co | None: ...

    async def find_many(
        self,
        take: int | None = None,
        skip: int | None = None,
        where: Mapping[str, object] | None = None,
        cursor: Mapping[str, object] | None = None,
        include: Mapping[str, object] | None = None,
        order: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
        distinct: Sequence[str] | None = None,
    ) -> Sequence[RowT_co]: ...

    async def create(self, data: Mapping[str, object], include: Mapping[str, object] | None = None) -> RowT_co: ...

    async def create_many(
        self, data: Sequence[Mapping[str, object]], *, skip_duplicates: bool | None = None
    ) -> int: ...

    async def upsert(
        self,
        where: Mapping[str, object],
        data: Mapping[str, object],
        include: Mapping[str, object] | None = None,
    ) -> RowT_co: ...

    async def update(
        self,
        data: Mapping[str, object],
        where: Mapping[str, object],
        include: Mapping[str, object] | None = None,
    ) -> RowT_co | None: ...

    async def update_many(self, data: Mapping[str, object], where: Mapping[str, object]) -> int: ...

    async def delete(
        self, where: Mapping[str, object], include: Mapping[str, object] | None = None
    ) -> RowT_co | None: ...

    async def delete_many(self, where: Mapping[str, object] | None = None) -> int: ...

    async def count(
        self,
        select: None = None,
        take: int | None = None,
        skip: int | None = None,
        where: Mapping[str, object] | None = None,
        cursor: Mapping[str, object] | None = None,
    ) -> int: ...

    async def group_by(
        self,
        by: Sequence[str],
        *,
        where: Mapping[str, object] | None = None,
        take: int | None = None,
        skip: int | None = None,
        order: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
        having: Mapping[str, object] | None = None,
        count: bool | Mapping[str, object] | None = None,
        sum: bool | Mapping[str, object] | None = None,
        avg: bool | Mapping[str, object] | None = None,
        min: bool | Mapping[str, object] | None = None,
        max: bool | Mapping[str, object] | None = None,
    ) -> Sequence[Mapping[str, object]]: ...


class PrismaRecord(Protocol):
    def dict(self) -> Mapping[str, object]: ...


class ReadOnlyTable(Protocol):
    async def find_many(self, *, where: Mapping[str, object]) -> Sequence[PrismaRecord]: ...


class SpendLinkedTable(Protocol[RowT_co]):
    async def find_many(self, *, where: Mapping[str, object]) -> Sequence[RowT_co]: ...

    async def update_many(self, *, where: Mapping[str, object], data: Mapping[str, object]) -> int: ...


class BatchTable(Protocol):
    def update(self, *, where: Mapping[str, object], data: Mapping[str, object]) -> None: ...

    def update_many(self, *, where: Mapping[str, object], data: Mapping[str, object]) -> None: ...


class PrismaBatch(Protocol):
    @property
    def litellm_verificationtoken(self) -> BatchTable: ...

    @property
    def litellm_usertable(self) -> BatchTable: ...

    @property
    def litellm_teamtable(self) -> BatchTable: ...

    @property
    def litellm_budgettable(self) -> BatchTable: ...

    @property
    def litellm_teammembership(self) -> BatchTable: ...

    @property
    def litellm_organizationtable(self) -> BatchTable: ...

    @property
    def litellm_tagtable(self) -> BatchTable: ...

    @property
    def litellm_endusertable(self) -> BatchTable: ...

    async def commit(self) -> None: ...
