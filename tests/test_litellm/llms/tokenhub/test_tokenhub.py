"""
Tests for TokenHub (Tencent Cloud) LLM provider.

Tests cover:
- Provider configuration and supported params
- Thinking parameter handling (enabled/disabled/adaptive)
- Reasoning effort parameter handling
- End-to-end completion call
- Provider routing and registration
"""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure local model cost map is used for tests that depend on model info
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import litellm
from litellm.llms.tokenhub.chat.transformation import TokenHubChatConfig
from litellm.llms.tokenhub.common_utils import TokenHubError
from litellm.utils import get_optional_params


class TestTokenHubConfig:
    def test_get_supported_params_with_reasoning_model(self):
        """Test that reasoning models include thinking and reasoning_effort params."""
        config = TokenHubChatConfig()
        # Use supports_reasoning mock to avoid dependency on model cost map loading
        with patch(
            "litellm.llms.tokenhub.chat.transformation.supports_reasoning",
            return_value=True,
        ):
            supported_params = config.get_supported_openai_params(
                model="tokenhub/deepseek-v4-pro"
            )
        assert "thinking" in supported_params
        assert "reasoning_effort" in supported_params
        assert "stream" in supported_params
        assert "tools" in supported_params

    def test_get_supported_params_non_reasoning_model(self):
        """Test that non-reasoning models do not include thinking params."""
        config = TokenHubChatConfig()
        with patch(
            "litellm.llms.tokenhub.chat.transformation.supports_reasoning",
            return_value=False,
        ):
            supported_params = config.get_supported_openai_params(
                model="tokenhub/some-basic-model"
            )
        assert "thinking" not in supported_params
        assert "reasoning_effort" not in supported_params
        # Base params should still be present
        assert "stream" in supported_params
        assert "tools" in supported_params

    def test_get_supported_params_base_params(self):
        """Test that base OpenAI params are always supported."""
        config = TokenHubChatConfig()
        with patch(
            "litellm.llms.tokenhub.chat.transformation.supports_reasoning",
            return_value=True,
        ):
            supported_params = config.get_supported_openai_params(
                model="tokenhub/deepseek-v4-pro"
            )
        expected_base_params = [
            "frequency_penalty",
            "function_call",
            "functions",
            "max_tokens",
            "max_completion_tokens",
            "response_format",
            "stop",
            "stream",
            "stream_options",
            "temperature",
            "tools",
            "tool_choice",
            "top_p",
        ]
        for param in expected_base_params:
            assert param in supported_params

    def test_thinking_enabled(self):
        """Test thinking enabled maps to extra_body."""
        config = TokenHubChatConfig()
        result = config.map_openai_params(
            non_default_params={"thinking": {"type": "enabled"}},
            optional_params={},
            model="tokenhub/deepseek-v4-pro",
            drop_params=False,
        )
        assert result == {"extra_body": {"thinking": {"type": "enabled"}}}

    def test_thinking_disabled(self):
        """Test thinking disabled maps to extra_body."""
        config = TokenHubChatConfig()
        result = config.map_openai_params(
            non_default_params={"thinking": {"type": "disabled"}},
            optional_params={},
            model="tokenhub/glm-5.2",
            drop_params=False,
        )
        assert result == {"extra_body": {"thinking": {"type": "disabled"}}}

    def test_thinking_adaptive(self):
        """Test thinking adaptive maps to extra_body."""
        config = TokenHubChatConfig()
        result = config.map_openai_params(
            non_default_params={"thinking": {"type": "adaptive"}},
            optional_params={},
            model="tokenhub/hy3-preview",
            drop_params=False,
        )
        assert result == {"extra_body": {"thinking": {"type": "adaptive"}}}

    def test_thinking_none_ignored(self):
        """Test that None thinking param is ignored."""
        config = TokenHubChatConfig()
        result = config.map_openai_params(
            non_default_params={"thinking": None},
            optional_params={},
            model="tokenhub/deepseek-v4-pro",
            drop_params=False,
        )
        assert result == {}

    def test_thinking_invalid_type_ignored(self):
        """Test that invalid thinking type is ignored."""
        config = TokenHubChatConfig()
        result = config.map_openai_params(
            non_default_params={"thinking": {"type": "invalid_type"}},
            optional_params={},
            model="tokenhub/deepseek-v4-pro",
            drop_params=False,
        )
        assert result == {}

    def test_thinking_non_dict_ignored(self):
        """Test that non-dict thinking value is ignored."""
        config = TokenHubChatConfig()
        result = config.map_openai_params(
            non_default_params={"thinking": "some_string"},
            optional_params={},
            model="tokenhub/deepseek-v4-pro",
            drop_params=False,
        )
        assert result == {}

    def test_reasoning_effort_valid(self):
        """Test valid reasoning_effort values are passed through."""
        config = TokenHubChatConfig()
        for effort in ("none", "low", "medium", "high", "xhigh"):
            result = config.map_openai_params(
                non_default_params={"reasoning_effort": effort},
                optional_params={},
                model="tokenhub/deepseek-v4-pro",
                drop_params=False,
            )
            assert result == {"extra_body": {"reasoning_effort": effort}}

    def test_reasoning_effort_invalid_ignored(self):
        """Test invalid reasoning_effort values are ignored."""
        config = TokenHubChatConfig()
        result = config.map_openai_params(
            non_default_params={"reasoning_effort": "invalid"},
            optional_params={},
            model="tokenhub/deepseek-v4-pro",
            drop_params=False,
        )
        assert result == {}

    def test_no_params(self):
        """Test empty params returns empty dict."""
        config = TokenHubChatConfig()
        result = config.map_openai_params(
            non_default_params={},
            optional_params={},
            model="tokenhub/deepseek-v4-pro",
            drop_params=False,
        )
        assert result == {}

    def test_e2e_get_optional_params(self):
        """Test end-to-end get_optional_params with tokenhub provider."""
        with patch(
            "litellm.llms.tokenhub.chat.transformation.supports_reasoning",
            return_value=True,
        ):
            e2e_mapped_params = get_optional_params(
                model="tokenhub/deepseek-v4-pro",
                custom_llm_provider="tokenhub",
                thinking={"type": "enabled"},
                drop_params=False,
            )
        assert "extra_body" in e2e_mapped_params
        assert e2e_mapped_params["extra_body"]["thinking"] == {"type": "enabled"}

    def test_e2e_completion_call(self):
        """Test end-to-end completion call with mocked client."""
        from openai import OpenAI

        from litellm import completion
        from litellm.types.utils import ModelResponse

        client = OpenAI(api_key="test_api_key")

        mock_raw_response = MagicMock()
        mock_raw_response.headers = {
            "x-request-id": "123",
            "openai-organization": "org-123",
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "99",
        }
        # Set parse on the return value of create() (which is what LiteLLM calls .parse() on)
        mock_raw_response.return_value.headers = mock_raw_response.headers
        mock_raw_response.return_value.parse.return_value = ModelResponse()

        with patch(
            "litellm.llms.tokenhub.chat.transformation.supports_reasoning",
            return_value=True,
        ), patch.object(
            client.chat.completions.with_raw_response, "create", mock_raw_response
        ) as mock_create:
            completion(
                model="tokenhub/deepseek-v4-pro",
                messages=[
                    {"role": "user", "content": "Hello"},
                ],
                stream=True,
                thinking={"type": "enabled"},
                client=client,
            )

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs
            assert "extra_body" in call_kwargs
            assert call_kwargs["extra_body"]["thinking"] == {"type": "enabled"}


class TestTokenHubError:
    def test_error_creation(self):
        """Test TokenHubError can be created with proper attributes."""
        error = TokenHubError(status_code=400, message="Bad request")
        assert error.status_code == 400
        assert error.message == "Bad request"

    def test_error_with_headers(self):
        """Test TokenHubError with custom headers."""
        import httpx

        headers = httpx.Headers({"x-request-id": "test-123"})
        error = TokenHubError(status_code=500, message="Server error", headers=headers)
        assert error.status_code == 500
        assert error.headers["x-request-id"] == "test-123"


class TestTokenHubProviderRouting:
    def test_provider_in_constants(self):
        """Test tokenhub is registered in provider lists."""
        from litellm.constants import openai_compatible_providers

        assert "tokenhub" in openai_compatible_providers

    def test_provider_enum(self):
        """Test tokenhub is in LlmProviders enum."""
        from litellm.types.utils import LlmProviders

        assert LlmProviders.TOKENHUB == "tokenhub"

    def test_tokenhub_in_local_model_cost_map(self):
        """Test tokenhub models exist in local model cost map."""
        from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

        model_cost_map = GetModelCostMap.load_local_model_cost_map()
        assert "tokenhub/deepseek-v4-pro" in model_cost_map
        assert (
            model_cost_map["tokenhub/deepseek-v4-pro"]["litellm_provider"] == "tokenhub"
        )

    def test_tokenhub_model_info(self):
        """Test tokenhub model info has correct fields."""
        from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

        model_cost_map = GetModelCostMap.load_local_model_cost_map()
        model_info = model_cost_map["tokenhub/deepseek-v4-pro"]
        assert model_info["max_input_tokens"] == 1000000
        assert model_info["max_output_tokens"] == 384000
        assert model_info["supports_function_calling"] is True
        assert model_info["supports_reasoning"] is True

    def test_get_llm_provider(self):
        """Test get_llm_provider correctly identifies tokenhub."""
        model, custom_llm_provider, dynamic_api_key, api_base = (
            litellm.get_llm_provider(model="tokenhub/deepseek-v4-pro")
        )
        assert custom_llm_provider == "tokenhub"
        assert api_base == "https://tokenhub.tencentmaas.com/v1"
