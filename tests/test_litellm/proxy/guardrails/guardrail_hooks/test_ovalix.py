"""
Unit tests for Ovalix guardrail types (OvalixGuardrailConfigModel) and config model resolution.
"""
from litellm.proxy.guardrails.guardrail_hooks.ovalix.ovalix import OvalixGuardrail


class TestOvalixGuardrailConfigModel:
    """Tests for OvalixGuardrailConfigModel from litellm.types.proxy.guardrails.guardrail_hooks.ovalix."""

    def test_config_model_ui_friendly_name(self):
        """Test that config model has correct UI friendly name."""
        from litellm.types.proxy.guardrails.guardrail_hooks.ovalix import (
            OvalixGuardrailConfigModel,
        )

        assert OvalixGuardrailConfigModel.ui_friendly_name() == "Ovalix Guardrail"

    def test_config_model_fields(self):
        """Test that config model has expected fields and default values."""
        from litellm.types.proxy.guardrails.guardrail_hooks.ovalix import (
            OvalixGuardrailConfigModel,
        )

        model = OvalixGuardrailConfigModel()

        assert model.tracker_api_base is None
        assert model.tracker_api_key is None
        assert model.application_id is None
        assert model.pre_checkpoint_id is None
        assert model.post_checkpoint_id is None

    def test_get_config_model(self):
        """Test get_config_model returns OvalixGuardrailConfigModel."""
        config_model = OvalixGuardrail.get_config_model()
        assert config_model is not None
        assert config_model.__name__ == "OvalixGuardrailConfigModel"
        assert hasattr(config_model, "ui_friendly_name")
        assert config_model.ui_friendly_name() == "Ovalix Guardrail"

    def test_config_model_with_all_fields_set(self):
        """Test construction with all optional fields set to explicit values."""
        from litellm.types.proxy.guardrails.guardrail_hooks.ovalix import (
            OvalixGuardrailConfigModel,
        )

        model = OvalixGuardrailConfigModel(
            tracker_api_base="https://tracker.ovalix.example",
            tracker_api_key="key-123",
            application_id="app-456",
            pre_checkpoint_id="pre-cp-1",
            post_checkpoint_id="post-cp-1",
        )

        assert model.tracker_api_base == "https://tracker.ovalix.example"
        assert model.tracker_api_key == "key-123"
        assert model.application_id == "app-456"
        assert model.pre_checkpoint_id == "pre-cp-1"
        assert model.post_checkpoint_id == "post-cp-1"

    def test_config_model_with_partial_fields(self):
        """Test construction with only a subset of fields set."""
        from litellm.types.proxy.guardrails.guardrail_hooks.ovalix import (
            OvalixGuardrailConfigModel,
        )

        model = OvalixGuardrailConfigModel(
            tracker_api_base="https://custom.tracker",
            application_id="app-only",
        )

        assert model.tracker_api_base == "https://custom.tracker"
        assert model.application_id == "app-only"
        assert model.tracker_api_key is None
        assert model.pre_checkpoint_id is None
        assert model.post_checkpoint_id is None

    def test_config_model_inherits_base_optional_params(self):
        """Test that model has optional_params from GuardrailConfigModel and defaults to None."""
        from litellm.types.proxy.guardrails.guardrail_hooks.base import (
            GuardrailConfigModel,
        )
        from litellm.types.proxy.guardrails.guardrail_hooks.ovalix import (
            OvalixGuardrailConfigModel,
        )

        model = OvalixGuardrailConfigModel()
        assert hasattr(model, "optional_params")
        assert model.optional_params is None
        assert isinstance(model, GuardrailConfigModel)

    def test_config_model_serialization_dump(self):
        """Test model_dump produces expected keys and values."""
        from litellm.types.proxy.guardrails.guardrail_hooks.ovalix import (
            OvalixGuardrailConfigModel,
        )

        model = OvalixGuardrailConfigModel(
            tracker_api_base="https://tracker.test",
            pre_checkpoint_id="pre-1",
        )
        data = model.model_dump()

        assert "tracker_api_base" in data
        assert "tracker_api_key" in data
        assert "application_id" in data
        assert "pre_checkpoint_id" in data
        assert "post_checkpoint_id" in data
        assert data["tracker_api_base"] == "https://tracker.test"
        assert data["pre_checkpoint_id"] == "pre-1"
        assert data["tracker_api_key"] is None

    def test_config_model_deserialization_from_dict(self):
        """Test model_validate builds instance from dict."""
        from litellm.types.proxy.guardrails.guardrail_hooks.ovalix import (
            OvalixGuardrailConfigModel,
        )

        payload = {
            "tracker_api_base": "https://from-dict.example",
            "tracker_api_key": "secret",
            "application_id": "app-dict",
            "pre_checkpoint_id": "pre-d",
            "post_checkpoint_id": "post-d",
        }
        model = OvalixGuardrailConfigModel.model_validate(payload)

        assert model.tracker_api_base == "https://from-dict.example"
        assert model.tracker_api_key == "secret"
        assert model.application_id == "app-dict"
        assert model.pre_checkpoint_id == "pre-d"
        assert model.post_checkpoint_id == "post-d"

    def test_config_model_deserialization_empty_dict(self):
        """Test model_validate with empty dict yields defaults."""
        from litellm.types.proxy.guardrails.guardrail_hooks.ovalix import (
            OvalixGuardrailConfigModel,
        )

        model = OvalixGuardrailConfigModel.model_validate({})

        assert model.tracker_api_base is None
        assert model.tracker_api_key is None
        assert model.application_id is None
        assert model.pre_checkpoint_id is None
        assert model.post_checkpoint_id is None

    def test_config_model_round_trip(self):
        """Test model_dump then model_validate preserves data."""
        from litellm.types.proxy.guardrails.guardrail_hooks.ovalix import (
            OvalixGuardrailConfigModel,
        )

        original = OvalixGuardrailConfigModel(
            tracker_api_base="https://round.trip",
            post_checkpoint_id="post-rt",
        )
        data = original.model_dump()
        restored = OvalixGuardrailConfigModel.model_validate(data)

        assert restored.tracker_api_base == original.tracker_api_base
        assert restored.tracker_api_key == original.tracker_api_key
        assert restored.application_id == original.application_id
        assert restored.pre_checkpoint_id == original.pre_checkpoint_id
        assert restored.post_checkpoint_id == original.post_checkpoint_id

    def test_config_model_has_expected_field_names(self):
        """Test that model defines all expected Ovalix config field names."""
        from litellm.types.proxy.guardrails.guardrail_hooks.ovalix import (
            OvalixGuardrailConfigModel,
        )

        expected = {
            "tracker_api_base",
            "tracker_api_key",
            "application_id",
            "pre_checkpoint_id",
            "post_checkpoint_id",
        }
        assert expected.issubset(OvalixGuardrailConfigModel.model_fields.keys())

    def test_config_model_field_descriptions_present(self):
        """Test that Ovalix-specific fields have non-empty descriptions."""
        from litellm.types.proxy.guardrails.guardrail_hooks.ovalix import (
            OvalixGuardrailConfigModel,
        )

        ovalix_fields = [
            "tracker_api_base",
            "tracker_api_key",
            "application_id",
            "pre_checkpoint_id",
            "post_checkpoint_id",
        ]
        for name in ovalix_fields:
            assert name in OvalixGuardrailConfigModel.model_fields
            desc = OvalixGuardrailConfigModel.model_fields[name].description
            assert desc is not None and len(desc) > 0
