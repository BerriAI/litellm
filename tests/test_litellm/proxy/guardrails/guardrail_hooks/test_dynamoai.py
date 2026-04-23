"""
Tests for DynamoAI guardrail registration and initialization.
"""

import os
from unittest.mock import patch

import pytest


class TestDynamoAIGuardrailRegistration:
    """Tests for DynamoAI guardrail registration in the guardrail system."""

    def test_supported_guardrail_enum_entry(self):
        """Test that DYNAMOAI is in SupportedGuardrailIntegrations enum."""
        from litellm.types.guardrails import SupportedGuardrailIntegrations

        assert hasattr(SupportedGuardrailIntegrations, "DYNAMOAI")
        assert SupportedGuardrailIntegrations.DYNAMOAI.value == "dynamoai"

    def test_initialize_guardrail_function_exists(self):
        """Test that initialize_guardrail function is properly exported."""
        from litellm.proxy.guardrails.guardrail_hooks.dynamoai import (
            guardrail_initializer_registry,
            initialize_guardrail,
        )

        assert initialize_guardrail is not None
        assert "dynamoai" in guardrail_initializer_registry

    def test_guardrail_class_registry_exists(self):
        """Test that guardrail_class_registry is properly exported."""
        from litellm.proxy.guardrails.guardrail_hooks.dynamoai import (
            guardrail_class_registry,
        )
        from litellm.proxy.guardrails.guardrail_hooks.dynamoai.dynamoai import (
            DynamoAIGuardrails,
        )

        assert "dynamoai" in guardrail_class_registry
        assert guardrail_class_registry["dynamoai"] == DynamoAIGuardrails

    def test_initialize_guardrail_creates_instance(self):
        """Test that initialize_guardrail creates a DynamoAIGuardrails instance."""
        from litellm.proxy.guardrails.guardrail_hooks.dynamoai import (
            initialize_guardrail,
        )
        from litellm.proxy.guardrails.guardrail_hooks.dynamoai.dynamoai import (
            DynamoAIGuardrails,
        )
        from litellm.types.guardrails import LitellmParams

        litellm_params = LitellmParams(
            guardrail="dynamoai",
            mode="pre_call",
            api_key="test-key",
            api_base="https://test.dynamo.ai",
        )

        guardrail = {
            "guardrail_name": "test-dynamoai-guard",
        }

        with patch(
            "litellm.logging_callback_manager.add_litellm_callback"
        ) as mock_add:
            result = initialize_guardrail(litellm_params, guardrail)

            assert isinstance(result, DynamoAIGuardrails)
            assert result.api_key == "test-key"
            assert result.api_base == "https://test.dynamo.ai"
            assert result.guardrail_name == "test-dynamoai-guard"
            mock_add.assert_called_once_with(result)

    def test_dynamoai_in_global_registry(self):
        """Test that dynamoai is discoverable in the global guardrail registry."""
        from litellm.proxy.guardrails.guardrail_registry import (
            guardrail_initializer_registry,
        )

        assert "dynamoai" in guardrail_initializer_registry
