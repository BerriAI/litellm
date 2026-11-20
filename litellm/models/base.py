"""
Base model class for domain models.
"""

from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Final, Protocol

from pydantic import ConfigDict
from typing_extensions import TypedDict

from litellm.litellm_core_utils.core_helpers import as_str_mapping, has_attributes
from litellm.types.llms.base import LiteLLMBaseModel


class _RecordFields(TypedDict, total=False, extra_items=object):
    pass


class _ModelDumpRecord(Protocol):
    def model_dump(self) -> Mapping[str, object]: ...


class _DictRecord(Protocol):
    def dict(self) -> Mapping[str, object]: ...


def _as_dict(record: object) -> Mapping[str, object] | None:
    fields: Final = as_str_mapping(record)
    return fields if isinstance(record, dict) else None


class DomainModel(LiteLLMBaseModel):
    """Base class for all domain models."""

    model_config = ConfigDict(
        from_attributes=True,
        protected_namespaces=(),
        extra="ignore",
    )

    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_db_record(cls, record: Mapping[str, object] | Iterable[tuple[str, object]] | None) -> "DomainModel":
        """Create a domain model from a database record."""
        if record is None:
            raise ValueError("Cannot create domain model from None record")
        fields: Final = _as_dict(record)
        if fields is not None:
            return cls(**_RecordFields(**fields))
        if has_attributes(record, _ModelDumpRecord, "model_dump") and callable(record.model_dump):
            return cls(**_RecordFields(**record.model_dump()))
        if has_attributes(record, _DictRecord, "dict") and callable(record.dict):
            return cls(**_RecordFields(**record.dict()))
        return cls(**_RecordFields(**dict(record)))

    def to_db_dict(self, exclude_unset: bool = False) -> dict[str, object]:
        """Convert domain model to a dictionary for database operations."""
        return self.model_dump(exclude_none=True, exclude_unset=exclude_unset)
