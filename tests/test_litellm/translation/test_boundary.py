"""The typed boundary: arktype-style parse over frozen pydantic models."""

from typing import List, Optional

import pytest
from pydantic import BaseModel, ConfigDict

from litellm.translation import boundary


class _Inner(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    value: int


class _Sample(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
    name: str
    items: Optional[List[_Inner]] = None


def test_parse_returns_validated_frozen_model() -> None:
    result = boundary.parse(_Sample, {"name": "x", "items": [{"value": 3}]})
    assert result.is_ok()
    model = result.ok
    assert model.name == "x"
    with pytest.raises(Exception):
        model.name = "mutated"  # type: ignore[misc]


def test_parse_accumulates_every_field_failure() -> None:
    result = boundary.parse(_Sample, {"name": 1, "items": [{"value": "no"}]})
    assert result.is_error()
    error = result.error
    assert error.tag == "boundary"
    assert len(error.boundary.failures) == 2
    assert "name" in error.boundary.summary
    assert "items.0.value" in error.boundary.summary


def test_unknown_fields_are_typed_unsupported_not_silently_dropped() -> None:
    result = boundary.parse(_Sample, {"name": "x", "mystery": 1, "other": 2})
    assert result.is_error()
    error = result.error
    assert error.tag == "unsupported"
    assert "mystery" in error.summary
    assert "other" in error.summary


def test_unknown_nested_field_is_unsupported() -> None:
    result = boundary.parse(_Sample, {"name": "x", "items": [{"value": 1, "extra": 2}]})
    assert result.is_error()
    assert result.error.tag == "unsupported"
    assert "items.0.extra" in result.error.summary


def test_as_plain_json_copies_and_never_aliases() -> None:
    source = {"a": [1, {"b": "c"}]}
    result = boundary.as_plain_json(source)
    assert result.is_ok()
    copied = result.ok
    assert copied == source
    source["a"].append("mutated")
    assert copied == {"a": [1, {"b": "c"}]}


def test_as_plain_json_rejects_non_json_leaves() -> None:
    result = boundary.as_plain_json({"a": object()})
    assert result.is_error()
    assert "not plain JSON" in result.error
