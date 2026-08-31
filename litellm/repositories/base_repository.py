"""
Base repository class with common functionality.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Final, Generic, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from litellm.repositories.prisma_protocols import TableActions

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class SupportsModelDump(Protocol):
    def model_dump(self) -> dict[str, object]: ...


@runtime_checkable
class SupportsDict(Protocol):
    def dict(self) -> dict[str, object]: ...


DbRecord = Mapping[str, object] | SupportsModelDump | SupportsDict | Sequence[tuple[str, object]]


def record_to_dict(record: DbRecord) -> Mapping[str, object]:
    """Project a database record into a mapping of column name to value."""
    if isinstance(record, SupportsModelDump):
        return record.model_dump()
    if isinstance(record, SupportsDict):
        return record.dict()
    if isinstance(record, Mapping):
        return record
    return {key: value for key, value in record}


class BaseRepository(ABC, Generic[T]):
    """Abstract base class for all repositories."""

    def __init__(self, prisma_client: Any):  # any-ok: PrismaClient is an untyped runtime wrapper
        self._prisma_client = prisma_client

    @property
    def prisma_client(self) -> Any:  # any-ok: PrismaClient is an untyped runtime wrapper
        if self._prisma_client is None:
            raise RuntimeError("No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys")
        return self._prisma_client

    @property
    @abstractmethod
    def table(self) -> TableActions[DbRecord]:
        """Return the Prisma table for this repository."""
        ...

    @property
    @abstractmethod
    def model_class(self) -> type[T]:
        """Return the domain model class for this repository."""
        ...

    def _to_model(self, record: DbRecord | None) -> T | None:
        """Convert a database record to a domain model."""
        if record is None:
            return None
        return self.model_class.model_validate(record_to_dict(record))

    def _to_model_list(self, records: Iterable[DbRecord | None]) -> list[T]:
        """Convert a list of database records to domain models."""
        return [model for record in records if record is not None and (model := self._to_model(record)) is not None]

    async def find_by_id(self, id_value: str, id_field: str = "id") -> T | None:
        """Find a record by its primary key."""
        record: Final = await self.table.find_unique(where={id_field: id_value})
        return self._to_model(record)

    async def find_many(
        self,
        where: Mapping[str, object] | None = None,
        skip: int | None = None,
        take: int | None = None,
        order: Mapping[str, str] | None = None,
    ) -> list[T]:
        """Find multiple records matching the criteria."""
        records: Final = await self.table.find_many(
            take=take,
            skip=skip,
            where=where or None,
            order=order or None,
        )
        return self._to_model_list(records)

    async def create(self, data: Mapping[str, object]) -> T:
        """Create a new record."""
        record: Final = await self.table.create(data=data)
        model: Final = self._to_model(record)
        assert model is not None
        return model

    async def update(self, id_value: str, data: Mapping[str, object], id_field: str = "id") -> T | None:
        """Update an existing record."""
        record: Final = await self.table.update(where={id_field: id_value}, data=data)
        return self._to_model(record)

    async def delete(self, id_value: str, id_field: str = "id") -> T | None:
        """Delete a record by its primary key."""
        record: Final = await self.table.delete(where={id_field: id_value})
        return self._to_model(record)

    async def count(self, where: Mapping[str, object] | None = None) -> int:
        """Count records matching the criteria."""
        return await self.table.count(where=where)

    async def exists(self, id_value: str, id_field: str = "id") -> bool:
        """Check if a record exists."""
        record: Final = await self.table.find_unique(where={id_field: id_value})
        return record is not None
