"""
Base repository class with common functionality.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Dict, Generic, List, Optional, Protocol, Tuple, Type, TypeVar, Union, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class SupportsModelDump(Protocol):
    def model_dump(self) -> Dict[str, object]: ...


@runtime_checkable
class SupportsDict(Protocol):
    def dict(self) -> Dict[str, object]: ...


DbRecord = Union[
    Mapping[str, object],
    SupportsModelDump,
    SupportsDict,
    Sequence[Tuple[str, object]],
]


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
    def table(self) -> Any:  # any-ok: Prisma table actions are reached through the untyped client wrapper
        """Return the Prisma table for this repository."""
        ...

    @property
    @abstractmethod
    def model_class(self) -> Type[T]:
        """Return the domain model class for this repository."""
        ...

    def _to_model(self, record: Optional[DbRecord]) -> Optional[T]:
        """Convert a database record to a domain model."""
        if record is None:
            return None
        return self.model_class.model_validate(record_to_dict(record))

    def _to_model_list(self, records: Iterable[Optional[DbRecord]]) -> List[T]:
        """Convert a list of database records to domain models."""
        return [model for record in records if record is not None and (model := self._to_model(record)) is not None]

    async def find_by_id(self, id_value: str, id_field: str = "id") -> Optional[T]:
        """Find a record by its primary key."""
        record = await self.table.find_unique(where={id_field: id_value})
        return self._to_model(record)

    async def find_many(
        self,
        where: Optional[Dict[str, Any]] = None,
        skip: Optional[int] = None,
        take: Optional[int] = None,
        order: Optional[Dict[str, str]] = None,
    ) -> List[T]:
        """Find multiple records matching the criteria."""
        kwargs: Dict[str, Any] = {}
        if where:
            kwargs["where"] = where
        if skip is not None:
            kwargs["skip"] = skip
        if take is not None:
            kwargs["take"] = take
        if order:
            kwargs["order"] = order

        records = await self.table.find_many(**kwargs)
        return self._to_model_list(records)

    async def create(self, data: Dict[str, Any]) -> T:
        """Create a new record."""
        record = await self.table.create(data=data)
        model = self._to_model(record)
        assert model is not None
        return model

    async def update(self, id_value: str, data: Dict[str, Any], id_field: str = "id") -> Optional[T]:
        """Update an existing record."""
        record = await self.table.update(where={id_field: id_value}, data=data)
        return self._to_model(record)

    async def delete(self, id_value: str, id_field: str = "id") -> Optional[T]:
        """Delete a record by its primary key."""
        record = await self.table.delete(where={id_field: id_value})
        return self._to_model(record)

    async def count(self, where: Optional[Dict[str, Any]] = None) -> int:
        """Count records matching the criteria."""
        return await self.table.count(where=where)

    async def exists(self, id_value: str, id_field: str = "id") -> bool:
        """Check if a record exists."""
        record = await self.table.find_unique(where={id_field: id_value})
        return record is not None
