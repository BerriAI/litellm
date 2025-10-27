"""Unit tests for Databricks Anthropic Messages Config.

Mock Data Explanation:
----------------------
These tests use mocked data (no real API calls) to verify the configuration logic:

1. URLs: Use generic 'test-databricks.net' format (works with any Databricks deployment)
2. API Keys: Use placeholder values like 'test_key' (real keys start with 'dapi')
3. Model Names: Use actual Anthropic model identifiers like 'claude-3-5-sonnet-20241022'
4. Parameters: Test standard Anthropic API parameters (max_tokens, temperature, tools, etc.)

The tests verify:
- URL construction (native endpoint doesn't add /v1/messages suffix)
- Bearer token authentication (Databricks format)
- Parameter support (Anthropic-compatible params)
- Provider recognition (databricks_anthropic/ prefix)
- Request/response pass-through (native protocol)
"""
import sys
import os
from unittest.mock import MagicMock
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
from litellm.llms.databricks.anthropic_messages.transformation import DatabricksAnthropicMessagesConfig


class TestDatabricksAnthropicMessagesConfig:
    
    def test_get_supported_params(self):
        """Test that all Anthropic-compatible parameters are supported.
        
        Mock data explanation:
        - Tests the list of parameter names supported by the API
        - max_tokens: Maximum response length
        - tools: Function calling support
        """
        config = DatabricksAnthropicMessagesConfig()
        params = config.get_supported_anthropic_messages_params(model="test")
        assert isinstance(params, list)
        assert "max_tokens" in params
        assert "tools" in params

    def test_get_complete_url(self):
        """Test URL construction for Databricks native Anthropic endpoint.
        
        Mock data explanation:
        - api_base: Databricks serving endpoint URL (works with any Databricks deployment)
        - Expected: URL should be returned as-is without /v1/messages suffix
        """
        config = DatabricksAnthropicMessagesConfig()
        api_base = "https://test-databricks.net/serving-endpoints/anthropic"
        result = config.get_complete_url(
            api_base=api_base,
            api_key="test_key",
            model="test_model",
            optional_params={},
            litellm_params={}
        )
        assert result == api_base

    def test_validate_environment(self):
        """Test Bearer token authentication setup.
        
        Mock data explanation:
        - headers: Empty dict that will be populated with auth header
        - test_key: Simulated Databricks API token (real tokens start with 'dapi')
        - Expected: Authorization header with 'Bearer {token}' format
        """
        config = DatabricksAnthropicMessagesConfig()
        headers = {}
        result_headers, result_api_base = config.validate_anthropic_messages_environment(
            headers=headers,
            model="test_model",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test_key"
        )
        assert "Authorization" in result_headers
        assert result_headers["Authorization"] == "Bearer test_key"

    def test_transform_request(self):
        """Test request parameter transformation.
        
        Mock data explanation:
        - params: Anthropic API parameters dict
        - max_tokens: 100 (example token limit for response)
        - Expected: Parameters passed through as-is for native endpoint
        """
        config = DatabricksAnthropicMessagesConfig()
        params = {"max_tokens": 100}
        result = config.transform_anthropic_messages_request(
            model="test_model",
            messages=[],
            anthropic_messages_optional_request_params=params,
            litellm_params={},
            headers={}
        )
        assert result == params

    def test_provider_recognition(self):
        """Test that databricks_anthropic/ prefix is correctly recognized.
        
        Mock data explanation:
        - model: 'databricks_anthropic/claude-3-5-sonnet-20241022'
          * Prefix 'databricks_anthropic/' triggers native Anthropic endpoint
          * Model name 'claude-3-5-sonnet-20241022' is the Anthropic model identifier
        - Expected: Provider identified as 'databricks'
        """
        from litellm import get_llm_provider
        model = "databricks_anthropic/claude-3-5-sonnet-20241022"
        provider, custom_llm_provider, _, _ = get_llm_provider(model=model)
        # With databricks_anthropic prefix, should recognize as databricks
        assert "databricks" in provider.lower() or custom_llm_provider == "databricks"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
