"""
Base model class for domain models.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict


class DomainModel(BaseModel):
    """Base class for all domain models."""

    model_config = ConfigDict(
        from_attributes=True,
        protected_namespaces=(),
        extra="ignore",
    )

    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @classmethod
    def from_db_record(cls, record: Any) -> "DomainModel":
        """Create a domain model from a database record."""
        if record is None:
            raise ValueError("Cannot create domain model from None record")
        if isinstance(record, dict):
            return cls(**record)
        if hasattr(record, "model_dump") and callable(record.model_dump):
            return cls(**record.model_dump())
        if hasattr(record, "dict") and callable(record.dict):
            return cls(**record.dict())
        return cls(**dict(record))

    def to_db_dict(self, exclude_unset: bool = False) -> Dict[str, Any]:
        """Convert domain model to a dictionary for database operations."""
        return self.model_dump(exclude_none=True, exclude_unset=exclude_unset)
