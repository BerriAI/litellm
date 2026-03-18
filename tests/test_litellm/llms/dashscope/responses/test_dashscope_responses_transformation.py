"""
Tests for DashScope Responses API transformation.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.llms.dashscope.responses.transformation import (
    DashScopeResponsesAPIConfig,
)
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


class TestDashScopeResponsesAPITransformation:
    """Test DashScope Responses API configuration and transformations."""

    def test_provider_config_registration(self):
        """Provider registry should return DashScopeResponsesAPIConfig."""
        config = ProviderConfigManager.get_provider_responses_api_config(
            model="dashscope/qwen-plus",
            provider=LlmProviders.DASHSCOPE,
        )

        assert config is not None, "Config should not be None for DashScope provider"
        assert isinstance(
            config, DashScopeResponsesAPIConfig
        ), f"Expected DashScopeResponsesAPIConfig, got {type(config)}"
        assert (
            config.custom_llm_provider == LlmProviders.DASHSCOPE
        ), "custom_llm_provider should be DASHSCOPE"

    def test_get_complete_url_default(self):
        """Default URL should point to DashScope compatible-mode endpoint."""
        config = DashScopeResponsesAPIConfig()
        url = config.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://dashscope.aliyuncs.com/compatible-mode/v1/responses"

    def test_get_complete_url_custom_base(self):
        """Custom api_base should be respected."""
        config = DashScopeResponsesAPIConfig()

        # Base ending with /v1
        url = config.get_complete_url(
            api_base="https://custom.example.com/v1", litellm_params={}
        )
        assert url == "https://custom.example.com/v1/responses"

        # Base already ending with /responses
        url = config.get_complete_url(
            api_base="https://custom.example.com/v1/responses", litellm_params={}
        )
        assert url == "https://custom.example.com/v1/responses"

        # Bare domain
        url = config.get_complete_url(
            api_base="https://custom.example.com", litellm_params={}
        )
        assert url == "https://custom.example.com/compatible-mode/v1/responses"

    @pytest.mark.parametrize(
        "litellm_params, expected_key",
        [
            ({"api_key": "dict-key"}, "dict-key"),
            (GenericLiteLLMParams(api_key="attr-key"), "attr-key"),
        ],
    )
    def test_validate_environment_uses_api_key(
        self, monkeypatch, litellm_params, expected_key
    ):
        """validate_environment should pull api key from params/env and attach headers."""
        config = DashScopeResponsesAPIConfig()

        monkeypatch.setattr(litellm, "api_key", None)
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

        headers = config.validate_environment(
            headers={}, model="dashscope/qwen-plus", litellm_params=litellm_params
        )

        assert headers.get("Authorization") == f"Bearer {expected_key}"
        assert headers.get("Content-Type") == "application/json"

    def test_validate_environment_from_env(self, monkeypatch):
        """validate_environment should fall back to DASHSCOPE_API_KEY env var."""
        config = DashScopeResponsesAPIConfig()

        monkeypatch.setattr(litellm, "api_key", None)
        monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")

        headers = config.validate_environment(
            headers={}, model="dashscope/qwen-plus", litellm_params={}
        )

        assert headers.get("Authorization") == "Bearer env-key"

    def test_validate_environment_raises_without_key(self, monkeypatch):
        """validate_environment should error when no key is available."""
        config = DashScopeResponsesAPIConfig()

        monkeypatch.setattr(litellm, "api_key", None)
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

        with pytest.raises(ValueError, match="DashScope API key is required"):
            config.validate_environment(
                headers={}, model="dashscope/qwen-plus", litellm_params={}
            )

    def test_supported_params(self):
        """Supported params should match documented DashScope surface."""
        config = DashScopeResponsesAPIConfig()
        supported = set(config.get_supported_openai_params("dashscope/qwen-plus"))

        expected = {
            "input",
            "model",
            "instructions",
            "max_output_tokens",
            "previous_response_id",
            "reasoning",
            "store",
            "stream",
            "temperature",
            "text",
            "tools",
            "tool_choice",
            "top_p",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        }

        assert supported == expected

    def test_map_openai_params_filters_unsupported(self):
        """map_openai_params should drop unsupported and metadata params."""
        config = DashScopeResponsesAPIConfig()
        params = ResponsesAPIOptionalRequestParams(
            temperature=0.7,
            metadata={"k": "v"},
        )

        mapped = config.map_openai_params(
            response_api_optional_params=params,
            model="dashscope/qwen-plus",
            drop_params=False,
        )

        assert mapped.get("temperature") == 0.7
        assert "metadata" not in mapped

    def test_extra_headers_merged(self, monkeypatch):
        """Extra headers passed to validate_environment should be merged."""
        config = DashScopeResponsesAPIConfig()

        monkeypatch.setattr(litellm, "api_key", None)
        monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

        headers = config.validate_environment(
            headers={"X-Custom": "value"},
            model="dashscope/qwen-plus",
            litellm_params={},
        )

        assert headers.get("X-Custom") == "value"
        assert headers.get("Authorization") == "Bearer test-key"
