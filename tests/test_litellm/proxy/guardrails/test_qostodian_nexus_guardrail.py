"""
Unit tests for Qostodian Nexus (by Qohash) integration.

Tests verify:
1. QostodianNexus can be instantiated with default and custom values
2. Qostodian Nexus is registered in SupportedGuardrailIntegrations
3. Guardrail initializer and class registries contain Qostodian Nexus
4. Configuration parameters are properly passed through
5. QostodianNexusConfigModel works correctly
"""

import os
import pytest
from unittest.mock import MagicMock


def test_qostodian_nexus_initialization_with_defaults():
    """Test QostodianNexus initializes with default values."""
    import os
    from unittest.mock import patch
    from litellm.proxy.guardrails.guardrail_hooks.qohash import QostodianNexus

    # Unset env var so the hardcoded default is used
    env = {k: v for k, v in os.environ.items() if k != "QOSTODIAN_NEXUS_API_BASE"}
    with patch.dict(os.environ, env, clear=True):
        guardrail = QostodianNexus()

    # Should use default api_base
    assert guardrail.api_base is not None
    assert "nexus:8800" in guardrail.api_base


def test_qostodian_nexus_initialization_with_custom_api_base():
    """Test QostodianNexus initializes with custom api_base."""
    from litellm.proxy.guardrails.guardrail_hooks.qohash import QostodianNexus

    custom_api_base = "http://custom-nexus:9000"
    guardrail = QostodianNexus(api_base=custom_api_base)

    assert custom_api_base in guardrail.api_base


def test_qostodian_nexus_in_supported_guardrail_integrations():
    """Test that Qostodian Nexus is registered in SupportedGuardrailIntegrations enum."""
    from litellm.types.guardrails import SupportedGuardrailIntegrations

    # Check enum contains QOSTODIAN_NEXUS
    assert hasattr(SupportedGuardrailIntegrations, "QOSTODIAN_NEXUS")
    assert SupportedGuardrailIntegrations.QOSTODIAN_NEXUS.value == "qostodian_nexus"

    # Check it's in the list of all values
    all_values = [e.value for e in SupportedGuardrailIntegrations]
    assert "qostodian_nexus" in all_values


def test_qostodian_nexus_in_guardrail_initializer_registry():
    """Test that Qostodian Nexus is registered in guardrail_initializer_registry."""
    from litellm.proxy.guardrails.guardrail_hooks.qohash import (
        guardrail_initializer_registry,
    )

    assert "qostodian_nexus" in guardrail_initializer_registry
    assert callable(guardrail_initializer_registry["qostodian_nexus"])


def test_qostodian_nexus_in_guardrail_class_registry():
    """Test that Qostodian Nexus is registered in guardrail_class_registry."""
    from litellm.proxy.guardrails.guardrail_hooks.qohash import (
        guardrail_class_registry,
        QostodianNexus,
    )

    assert "qostodian_nexus" in guardrail_class_registry
    assert guardrail_class_registry["qostodian_nexus"] == QostodianNexus


def test_qostodian_nexus_config_model_initialization():
    """Test QostodianNexusConfigModel can be instantiated."""
    from litellm.types.proxy.guardrails.guardrail_hooks.qohash import (
        QostodianNexusConfigModel,
    )

    config = QostodianNexusConfigModel(
        api_base="http://test:8800",
    )

    assert config.api_base == "http://test:8800"


def test_qostodian_nexus_config_model_defaults():
    """Test QostodianNexusConfigModel uses correct defaults."""
    from litellm.types.proxy.guardrails.guardrail_hooks.qohash import (
        QostodianNexusConfigModel,
    )

    config = QostodianNexusConfigModel()

    assert config.api_base is None


def test_qostodian_nexus_config_model_ui_friendly_name():
    """Test QostodianNexusConfigModel returns correct UI friendly name."""
    from litellm.types.proxy.guardrails.guardrail_hooks.qohash import (
        QostodianNexusConfigModel,
    )

    ui_name = QostodianNexusConfigModel.ui_friendly_name()
    assert ui_name == "Qostodian Nexus"


def test_qostodian_nexus_initializer_function():
    """Test the initialize_guardrail function."""
    from litellm.proxy.guardrails.guardrail_hooks.qohash import initialize_guardrail
    from litellm.types.guardrails import LitellmParams, Guardrail
    from unittest.mock import patch

    # Mock litellm.logging_callback_manager
    with patch("litellm.logging_callback_manager") as mock_manager:
        mock_manager.add_litellm_callback = MagicMock()

        # Create test params
        litellm_params = LitellmParams(
            guardrail="qostodian_nexus",
            mode="pre_call",
            api_base="http://test:8800",
            default_on=True,
        )

        guardrail_config: Guardrail = {"guardrail_name": "test-qostodian-nexus"}

        # Call initializer
        result = initialize_guardrail(litellm_params, guardrail_config)

        # Verify callback was added
        mock_manager.add_litellm_callback.assert_called_once()

        # Verify returned instance has correct properties
        assert result is not None
        assert "test:8800" in result.api_base


def test_qostodian_nexus_inherits_from_generic_guardrail_api():
    """Test that QostodianNexus inherits from GenericGuardrailAPI."""
    from litellm.proxy.guardrails.guardrail_hooks.qohash import QostodianNexus
    from litellm.proxy.guardrails.guardrail_hooks.generic_guardrail_api.generic_guardrail_api import (
        GenericGuardrailAPI,
    )

    assert issubclass(QostodianNexus, GenericGuardrailAPI)


def test_qostodian_nexus_guardrail_name_constant():
    """Test that GUARDRAIL_NAME constant is defined correctly."""
    from litellm.proxy.guardrails.guardrail_hooks.qohash.qohash import GUARDRAIL_NAME

    assert GUARDRAIL_NAME == "qostodian_nexus"


def test_qostodian_nexus_get_config_model():
    """Test that QostodianNexus returns the correct config model."""
    from litellm.proxy.guardrails.guardrail_hooks.qohash import QostodianNexus
    from litellm.types.proxy.guardrails.guardrail_hooks.qohash import (
        QostodianNexusConfigModel,
    )

    config_model = QostodianNexus.get_config_model()

    assert config_model is not None
    assert config_model == QostodianNexusConfigModel


def test_qostodian_nexus_env_vars():
    """Test that QOSTODIAN_NEXUS_API_BASE env var is picked up correctly."""
    import os
    from unittest.mock import patch
    from litellm.proxy.guardrails.guardrail_hooks.qohash import QostodianNexus

    with patch.dict(os.environ, {"QOSTODIAN_NEXUS_API_BASE": "http://new-api:8800"}):
        guardrail = QostodianNexus()
        assert "new-api:8800" in guardrail.api_base


def test_qostodian_nexus_config_model_field_descriptions():
    """Test that QostodianNexusConfigModel has correct field descriptions."""
    from litellm.types.proxy.guardrails.guardrail_hooks.qohash import (
        QostodianNexusConfigModel,
    )

    # Check that field descriptions mention the correct env vars
    api_base_field = QostodianNexusConfigModel.model_fields["api_base"]
    assert "QOSTODIAN_NEXUS_API_BASE" in api_base_field.description


def test_qostodian_nexus_unified_detection():
    """
    Test that QostodianNexus is properly detected by LiteLLM's unified guardrail system.

    This verifies the fix for the detection bug where QostodianNexus wasn't being
    recognized because apply_guardrail was only inherited, not in the class's own __dict__.
    """
    from litellm.proxy.guardrails.guardrail_hooks.qohash import QostodianNexus

    # Create an instance (this is how LiteLLM uses it)
    instance = QostodianNexus(api_base="http://test:8800")

    # Test the exact detection logic used in litellm/proxy/utils.py:868
    # use_unified = "apply_guardrail" in type(callback).__dict__
    use_unified = "apply_guardrail" in type(instance).__dict__

    # Should be detected as using unified guardrail system
    assert use_unified is True, (
        "QostodianNexus should be detected by unified guardrail system. "
        "The apply_guardrail method must be present in QostodianNexus.__dict__"
    )

    # Also verify the method is callable
    assert hasattr(instance, "apply_guardrail")
    assert callable(instance.apply_guardrail)


def test_qostodian_nexus_builtin_extra_headers():
    """Test that QostodianNexus includes built-in x-qostodian-nexus-identifiers-* headers."""
    from litellm.proxy.guardrails.guardrail_hooks.qohash import QostodianNexus

    instance = QostodianNexus()

    expected_headers = [
        "x-qostodian-nexus-identifiers-trace",
        "x-qostodian-nexus-identifiers-source",
        "x-qostodian-nexus-identifiers-container",
        "x-qostodian-nexus-identifiers-identity",
    ]

    for header in expected_headers:
        assert header in instance.extra_headers, (
            f"Expected built-in header '{header}' to be in extra_headers"
        )


def test_qostodian_nexus_extra_headers_merged():
    """Test that caller-supplied extra_headers are merged with built-in headers."""
    from litellm.proxy.guardrails.guardrail_hooks.qohash import QostodianNexus

    custom_header = "x-custom-correlation-id"
    instance = QostodianNexus(extra_headers=[custom_header])

    # Built-in headers should be present
    assert "x-qostodian-nexus-identifiers-trace" in instance.extra_headers
    # Custom header should also be present
    assert custom_header in instance.extra_headers
    # No duplicates
    assert len(instance.extra_headers) == len(set(instance.extra_headers))
