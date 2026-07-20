import logging
from typing import Any, Dict, Optional
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ValidationError

from litellm.models.managed_files import LiteLLM_ManagedVectorStoresTable
from litellm.proxy.common_utils.cache_pydantic_utils import CacheCodec


class _SampleModel(BaseModel):
    name: str
    count: Optional[int] = None


class _SampleSubModel(_SampleModel):
    pass


class _RequiredNullableModel(BaseModel):
    id: str
    budget_table: Optional[Dict[str, Any]]


class TestCacheCodecSerialize:
    def test_without_model_type_base_model_dumped_json_safe(self):
        m = _SampleModel(name="a", count=1)
        out = CacheCodec.serialize(m)
        assert out == {"name": "a", "count": 1}

    def test_without_model_type_dict_json_sanitized(self):
        from datetime import datetime, timezone

        d = {
            "name": "x",
            "budget_reset_at": datetime(2026, 8, 1, tzinfo=timezone.utc),
        }
        out = CacheCodec.serialize(d)
        assert out == {
            "name": "x",
            "budget_reset_at": "2026-08-01T00:00:00+00:00",
        }
        assert out is not d

    def test_without_model_type_plain_dict_values_preserved(self):
        d = {"name": "x", "count": 1}
        out = CacheCodec.serialize(d)
        assert out == d
        assert out is not d

    def test_without_model_type_primitive_unchanged(self):
        assert CacheCodec.serialize(42) == 42

    def test_with_model_type_dict_validated_and_dumped(self):
        out = CacheCodec.serialize({"name": "b", "count": 2}, model_type=_SampleModel)
        assert out == {"name": "b", "count": 2}

    def test_with_model_type_base_model_validated_and_dumped(self):
        m = _SampleModel(name="c", count=None)
        out = CacheCodec.serialize(m, model_type=_SampleModel)
        assert out == {"name": "c", "count": None}

    def test_with_model_type_none_field_preserved_on_dump(self):
        out = CacheCodec.serialize({"name": "d"}, model_type=_SampleModel)
        assert out == {"name": "d", "count": None}
        assert "count" in out

    def test_with_model_type_non_dict_non_model_passthrough(self):
        assert CacheCodec.serialize("raw", model_type=_SampleModel) == "raw"

    def test_with_model_type_invalid_dict_raises(self):
        with pytest.raises(ValidationError):
            CacheCodec.serialize({"count": 1}, model_type=_SampleModel)

    def test_with_model_type_already_correct_instance_skips_revalidation(self):
        """Fast-path: value is already model_type — model_validate must NOT be called."""
        m = _SampleModel(name="fast", count=7)
        with patch.object(_SampleModel, "model_validate", wraps=_SampleModel.model_validate) as mock_validate:
            out = CacheCodec.serialize(m, model_type=_SampleModel)
        assert out == {"name": "fast", "count": 7}
        mock_validate.assert_not_called()

    def test_with_model_type_subclass_instance_skips_revalidation(self):
        """Subclass is isinstance of base → should also take the fast path."""
        sub = _SampleSubModel(name="sub", count=2)
        with patch.object(_SampleModel, "model_validate", wraps=_SampleModel.model_validate) as mock_validate:
            out = CacheCodec.serialize(sub, model_type=_SampleModel)
        assert out == {"name": "sub", "count": 2}
        mock_validate.assert_not_called()

    def test_with_model_type_dict_input_goes_through_model_validate(self):
        """A dict value (not yet an instance) must still go through model_validate."""
        raw = {"name": "via-dict", "count": 5}
        with patch.object(
            _SampleModel, "model_validate", wraps=_SampleModel.model_validate
        ) as mock_validate:
            out = CacheCodec.serialize(raw, model_type=_SampleModel)
        assert out == {"name": "via-dict", "count": 5}
        mock_validate.assert_called_once()

    def test_with_model_type_incompatible_model_raises_validation_error(self):
        """Passing a BaseModel whose fields don't satisfy model_type's required fields raises.

        _IncompatibleModel only has `foo: int`, so when Pydantic v2 extracts its
        data and validates it against _SampleModel (which requires `name: str`),
        a ValidationError is raised.
        """

        class _IncompatibleModel(BaseModel):
            foo: int  # missing required 'name' field of _SampleModel

        with pytest.raises(ValidationError):
            CacheCodec.serialize(_IncompatibleModel(foo=1), model_type=_SampleModel)


class TestCacheCodecDeserialize:
    def test_none_returns_none(self):
        assert CacheCodec.deserialize(None, _SampleModel) is None

    def test_dict_validates_to_model(self):
        m = CacheCodec.deserialize({"name": "e", "count": 3}, _SampleModel)
        assert isinstance(m, _SampleModel)
        assert m.name == "e"
        assert m.count == 3

    def test_instance_same_type_returned_as_is(self):
        original = _SampleModel(name="f")
        m = CacheCodec.deserialize(original, _SampleModel)
        assert m is original

    def test_subclass_instance_accepted(self):
        sub = _SampleSubModel(name="g")
        m = CacheCodec.deserialize(sub, _SampleModel)
        assert m is sub

    def test_wrong_type_returns_none(self):
        assert CacheCodec.deserialize("not-a-dict", _SampleModel) is None

    def test_invalid_dict_returns_none_and_logs_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
            out = CacheCodec.deserialize({"count": 1}, _SampleModel)
        assert out is None
        assert any(
            "CacheCodec.deserialize" in r.message and "_SampleModel" in r.message
            for r in caplog.records
            if r.levelno >= logging.WARNING
        ), f"Expected deserialize validation warning. Records: {[r.message for r in caplog.records]}"


class TestCacheCodecRoundTripPreservesNoneFields:
    def test_none_value_kept_as_null_not_dropped(self):
        out = CacheCodec.serialize(
            _RequiredNullableModel(id="x", budget_table=None),
            model_type=_RequiredNullableModel,
        )
        assert out == {"id": "x", "budget_table": None}
        assert "budget_table" in out

    def test_required_nullable_none_field_survives_round_trip(self):
        original = _RequiredNullableModel(id="x", budget_table=None)
        wire = CacheCodec.serialize(original, model_type=_RequiredNullableModel)
        restored = CacheCodec.deserialize(wire, model_type=_RequiredNullableModel)
        assert restored == original

    def test_managed_vector_store_row_round_trips_with_optional_fields_none(self):
        vs = LiteLLM_ManagedVectorStoresTable(
            vector_store_id="vs_1",
            custom_llm_provider="openai",
            vector_store_name=None,
            vector_store_description=None,
            vector_store_metadata=None,
            created_at=None,
            updated_at=None,
            litellm_credential_name=None,
            litellm_params=None,
            team_id=None,
            user_id=None,
        )
        wire = CacheCodec.serialize(vs, model_type=LiteLLM_ManagedVectorStoresTable)
        assert wire.get("vector_store_name", "MISSING") is None
        assert wire.get("team_id", "MISSING") is None
        restored = CacheCodec.deserialize(wire, model_type=LiteLLM_ManagedVectorStoresTable)
        assert restored == vs
