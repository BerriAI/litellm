"""
Unit tests for Qohash guardrail integration.

Tests verify:
1. QohashGuardrail can be instantiated with default and custom values
2. Qohash is registered in SupportedGuardrailIntegrations
3. Guardrail initializer and class registries contain Qohash
4. Configuration parameters are properly passed through
5. QohashGuardrailConfigModel works correctly
"""

import os
import pytest
from unittest.mock import MagicMock


def test_qohash_guardrail_initialization_with_defaults():
    """Test QohashGuardrail initializes with default values."""
    from litellm.proxy.guardrails.guardrail_hooks.qohash import QohashGuardrail

    guardrail = QohashGuardrail()

    # Should use default api_base
    assert guardrail.api_base is not None
    assert "qaigs:8800" in guardrail.api_base


def test_qohash_guardrail_initialization_with_custom_api_base():
    """Test QohashGuardrail initializes with custom api_base."""
    from litellm.proxy.guardrails.guardrail_hooks.qohash import QohashGuardrail

    custom_api_base = "http://custom-guardrail:9000"
    guardrail = QohashGuardrail(api_base=custom_api_base)

    assert custom_api_base in guardrail.api_base


def test_qohash_guardrail_initialization_with_all_params():
    """Test QohashGuardrail initializes with all custom parameters."""
    from litellm.proxy.guardrails.guardrail_hooks.qohash import QohashGuardrail

    guardrail = QohashGuardrail(
        api_base="http://test-guardrail:8888",
        api_key="test-key-123",
    )

    assert "test-guardrail:8888" in guardrail.api_base
    # api_key is stored in headers as x-api-key, not as a direct attribute
    assert guardrail.headers.get("x-api-key") == "test-key-123"


def test_qohash_in_supported_guardrail_integrations():
    """Test that Qohash is registered in SupportedGuardrailIntegrations enum."""
    from litellm.types.guardrails import SupportedGuardrailIntegrations

    # Check enum contains QOHASH
    assert hasattr(SupportedGuardrailIntegrations, "QOHASH")
    assert SupportedGuardrailIntegrations.QOHASH.value == "qohash_qaigs"

    # Check it's in the list of all values
    all_values = [e.value for e in SupportedGuardrailIntegrations]
    assert "qohash_qaigs" in all_values


def test_qohash_in_guardrail_initializer_registry():
    """Test that Qohash is registered in guardrail_initializer_registry."""
    from litellm.proxy.guardrails.guardrail_hooks.qohash import (
        guardrail_initializer_registry,
    )

    assert "qohash_qaigs" in guardrail_initializer_registry
    assert callable(guardrail_initializer_registry["qohash_qaigs"])


def test_qohash_in_guardrail_class_registry():
    """Test that Qohash is registered in guardrail_class_registry."""
    from litellm.proxy.guardrails.guardrail_hooks.qohash import (
        guardrail_class_registry,
    )
    from litellm.proxy.guardrails.guardrail_hooks.qohash import QohashGuardrail

    assert "qohash_qaigs" in guardrail_class_registry
    assert guardrail_class_registry["qohash_qaigs"] == QohashGuardrail


def test_qohash_config_model_initialization():
    """Test QohashGuardrailConfigModel can be instantiated."""
    from litellm.types.proxy.guardrails.guardrail_hooks.qohash import (
        QohashGuardrailConfigModel,
    )

    config = QohashGuardrailConfigModel(
        api_base="http://test:8800",
        api_key="test-key",
    )

    assert config.api_base == "http://test:8800"
    assert config.api_key == "test-key"


def test_qohash_config_model_defaults():
    """Test QohashGuardrailConfigModel uses correct defaults."""
    from litellm.types.proxy.guardrails.guardrail_hooks.qohash import (
        QohashGuardrailConfigModel,
    )

    config = QohashGuardrailConfigModel()

    assert config.api_base is None
    assert config.api_key is None


def test_qohash_config_model_ui_friendly_name():
    """Test QohashGuardrailConfigModel returns correct UI friendly name."""
    from litellm.types.proxy.guardrails.guardrail_hooks.qohash import (
        QohashGuardrailConfigModel,
    )

    ui_name = QohashGuardrailConfigModel.ui_friendly_name()
    assert ui_name == "Qohash AI Guardrail Server"


def test_qohash_guardrail_initializer_function():
    """Test the initialize_guardrail function."""
    from litellm.proxy.guardrails.guardrail_hooks.qohash import initialize_guardrail
    from litellm.types.guardrails import LitellmParams, Guardrail
    from unittest.mock import patch

    # Mock litellm.logging_callback_manager
    with patch("litellm.logging_callback_manager") as mock_manager:
        mock_manager.add_litellm_callback = MagicMock()

        # Create test params
        litellm_params = LitellmParams(
            guardrail="qohash_qaigs",
            mode="pre_call",
            api_base="http://test:8800",
            api_key="test-key",
            default_on=True,
        )

        guardrail_config: Guardrail = {"guardrail_name": "test-qohash"}

        # Call initializer
        result = initialize_guardrail(litellm_params, guardrail_config)

        # Verify callback was added
        mock_manager.add_litellm_callback.assert_called_once()

        # Verify returned guardrail has correct properties
        assert result is not None
        assert "test:8800" in result.api_base


def test_qohash_inherits_from_generic_guardrail_api():
    """Test that QohashGuardrail inherits from GenericGuardrailAPI."""
    from litellm.proxy.guardrails.guardrail_hooks.qohash import QohashGuardrail
    from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api.generic_guardrail_api import (
        GenericGuardrailAPI,
    )

    assert issubclass(QohashGuardrail, GenericGuardrailAPI)


def test_qohash_guardrail_name_constant():
    """Test that GUARDRAIL_NAME constant is defined correctly."""
    from litellm.proxy.guardrails.guardrail_hooks.qohash.qohash import GUARDRAIL_NAME

    assert GUARDRAIL_NAME == "qohash_qaigs"


def test_qohash_get_config_model():
    """Test that QohashGuardrail returns the correct config model."""
    from litellm.proxy.guardrails.guardrail_hooks.qohash import QohashGuardrail
    from litellm.types.proxy.guardrails.guardrail_hooks.qohash import (
        QohashGuardrailConfigModel,
    )

    config_model = QohashGuardrail.get_config_model()

    assert config_model is not None
    assert config_model == QohashGuardrailConfigModel


def test_qohash_guardrail_env_vars():
    """Test that QAIGS_* env vars are picked up correctly."""
    import os
    from unittest.mock import patch
    from litellm.proxy.guardrails.guardrail_hooks.qohash import QohashGuardrail

    with patch.dict(os.environ, {"QAIGS_API_BASE": "http://new-api:8800"}):
        guardrail = QohashGuardrail()
        assert "new-api:8800" in guardrail.api_base

    with patch.dict(os.environ, {"QAIGS_API_KEY": "new-key"}):
        guardrail = QohashGuardrail()
        assert guardrail.headers.get("x-api-key") == "new-key"


def test_qohash_config_model_field_descriptions():
    """Test that QohashGuardrailConfigModel has correct field descriptions."""
    from litellm.types.proxy.guardrails.guardrail_hooks.qohash import (
        QohashGuardrailConfigModel,
    )

    # Check that field descriptions mention the correct env vars
    api_key_field = QohashGuardrailConfigModel.model_fields["api_key"]
    assert "QAIGS_API_KEY" in api_key_field.description

    api_base_field = QohashGuardrailConfigModel.model_fields["api_base"]
    assert "QAIGS_API_BASE" in api_base_field.description


def test_qohash_guardrail_unified_detection():
    """
    Test that QohashGuardrail is properly detected by LiteLLM's unified guardrail system.

    This verifies the fix for the detection bug where QohashGuardrail wasn't being
    recognized because apply_guardrail was only inherited, not in the class's own __dict__.
    """
    from litellm.proxy.guardrails.guardrail_hooks.qohash import QohashGuardrail

    # Create an instance (this is how LiteLLM uses it)
    guardrail = QohashGuardrail(api_base="http://test:8800", api_key="test")

    # Test the exact detection logic used in litellm/proxy/utils.py:868
    # use_unified = "apply_guardrail" in type(callback).__dict__
    use_unified = "apply_guardrail" in type(guardrail).__dict__

    # Should be detected as using unified guardrail system
    assert use_unified is True, (
        "QohashGuardrail should be detected by unified guardrail system. "
        "The apply_guardrail method must be present in QohashGuardrail.__dict__"
    )

    # Also verify the method is callable
    assert hasattr(guardrail, "apply_guardrail")
    assert callable(guardrail.apply_guardrail)
