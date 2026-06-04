"""
Base repository class with common functionality.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

from litellm.backend.models.base import DomainModel

T = TypeVar("T", bound=DomainModel)


class BaseRepository(ABC, Generic[T]):
    """Abstract base class for all repositories."""

    def __init__(self, prisma_client: Any):
        self._prisma_client = prisma_client

    @property
    def prisma_client(self) -> Any:
        if self._prisma_client is None:
            raise RuntimeError(
                "No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys"
            )
        return self._prisma_client

    @property
    @abstractmethod
    def table(self) -> Any:
        """Return the Prisma table for this repository."""
        ...

    @property
    @abstractmethod
    def model_class(self) -> Type[T]:
        """Return the domain model class for this repository."""
        ...

    def _to_model(self, record: Any) -> Optional[T]:
        """Convert a database record to a domain model."""
        if record is None:
            return None
        return self.model_class.from_db_record(record)

    def _to_model_list(self, records: List[Any]) -> List[T]:
        """Convert a list of database records to domain models."""
        return [self._to_model(r) for r in records if r is not None]

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
        return self._to_model(record)

    async def update(
        self, id_value: str, data: Dict[str, Any], id_field: str = "id"
    ) -> Optional[T]:
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
