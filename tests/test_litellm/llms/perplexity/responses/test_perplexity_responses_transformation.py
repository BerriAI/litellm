"""
Tests for Perplexity Responses API transformation

Tests the PerplexityResponsesConfig class that handles Perplexity-specific
transformations for the Agent API (Responses API).

Source: litellm/llms/perplexity/responses/transformation.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.perplexity.responses.transformation import PerplexityResponsesConfig
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


class TestPerplexityResponsesTransformation:
    """Test Perplexity Responses API configuration and transformations"""

    def test_function_tool_passthrough(self):
        """Function tools with name/description/parameters are preserved"""
        config = PerplexityResponsesConfig()

        params = ResponsesAPIOptionalRequestParams(
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                },
                            },
                        },
                    },
                }
            ]
        )

        result = config.map_openai_params(
            response_api_optional_params=params,
            model="perplexity/openai/gpt-5.2",
            drop_params=False,
        )

        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["function"]["name"] == "get_weather"
        assert (
            result["tools"][0]["function"]["description"] == "Get the current weather"
        )
        assert "parameters" in result["tools"][0]["function"]

    def test_web_search_tool_passthrough(self):
        """web_search tools are passed through unchanged"""
        config = PerplexityResponsesConfig()

        params = ResponsesAPIOptionalRequestParams(tools=[{"type": "web_search"}])

        result = config.map_openai_params(
            response_api_optional_params=params,
            model="perplexity/openai/gpt-5.2",
            drop_params=False,
        )

        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "web_search"

    def test_fetch_url_tool_passthrough(self):
        """fetch_url tools are passed through"""
        config = PerplexityResponsesConfig()

        params = ResponsesAPIOptionalRequestParams(tools=[{"type": "fetch_url"}])

        result = config.map_openai_params(
            response_api_optional_params=params,
            model="perplexity/openai/gpt-5.2",
            drop_params=False,
        )

        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "fetch_url"

    def test_mixed_tools_function_and_web_search(self):
        """Mixed function and web_search tools are transformed correctly"""
        config = PerplexityResponsesConfig()

        params = ResponsesAPIOptionalRequestParams(
            tools=[
                {"type": "web_search"},
                {
                    "type": "function",
                    "function": {
                        "name": "custom_tool",
                        "description": "A custom tool",
                        "parameters": {"type": "object"},
                    },
                },
            ]
        )

        result = config.map_openai_params(
            response_api_optional_params=params,
            model="perplexity/openai/gpt-5.2",
            drop_params=False,
        )

        assert len(result["tools"]) == 2
        assert result["tools"][0]["type"] == "web_search"
        assert result["tools"][1]["type"] == "function"
        assert result["tools"][1]["function"]["name"] == "custom_tool"

    def test_tool_choice_mapping(self):
        """tool_choice passes through"""
        config = PerplexityResponsesConfig()

        params = ResponsesAPIOptionalRequestParams(
            tool_choice="required", temperature=0.7
        )

        result = config.map_openai_params(
            response_api_optional_params=params,
            model="perplexity/openai/gpt-5.2",
            drop_params=False,
        )

        assert result.get("tool_choice") == "required"

    def test_parallel_tool_calls(self):
        """parallel_tool_calls passes through"""
        config = PerplexityResponsesConfig()

        params = ResponsesAPIOptionalRequestParams(
            parallel_tool_calls=True, temperature=0.7
        )

        result = config.map_openai_params(
            response_api_optional_params=params,
            model="perplexity/openai/gpt-5.2",
            drop_params=False,
        )

        assert result.get("parallel_tool_calls") is True

    def test_max_tool_calls_mapping(self):
        """max_tool_calls passes through"""
        config = PerplexityResponsesConfig()

        params = ResponsesAPIOptionalRequestParams(max_tool_calls=5, temperature=0.7)

        result = config.map_openai_params(
            response_api_optional_params=params,
            model="perplexity/openai/gpt-5.2",
            drop_params=False,
        )

        assert result.get("max_tool_calls") == 5

    def test_text_passthrough(self):
        """text param passes through as-is (Perplexity accepts Open Responses format directly)"""
        config = PerplexityResponsesConfig()

        text_value = {
            "format": {
                "type": "json_schema",
                "name": "weather_response",
                "schema": {
                    "type": "object",
                    "properties": {"temp": {"type": "number"}},
                },
                "strict": True,
            }
        }

        params = ResponsesAPIOptionalRequestParams(
            text=text_value,
            temperature=0.7,
        )

        result = config.map_openai_params(
            response_api_optional_params=params,
            model="perplexity/openai/gpt-5.2",
            drop_params=False,
        )

        assert "text" in result
        assert result["text"] == text_value
        assert "response_format" not in result

    def test_previous_response_id(self):
        """previous_response_id passes through"""
        config = PerplexityResponsesConfig()

        params = ResponsesAPIOptionalRequestParams(
            previous_response_id="resp_abc123",
            temperature=0.7,
        )

        result = config.map_openai_params(
            response_api_optional_params=params,
            model="perplexity/openai/gpt-5.2",
            drop_params=False,
        )

        assert result.get("previous_response_id") == "resp_abc123"

    def test_store_background_truncation(self):
        """Lifecycle params pass through"""
        config = PerplexityResponsesConfig()

        params = ResponsesAPIOptionalRequestParams(
            store=True,
            background=False,
            truncation="auto",
            temperature=0.7,
        )

        result = config.map_openai_params(
            response_api_optional_params=params,
            model="perplexity/openai/gpt-5.2",
            drop_params=False,
        )

        assert result.get("store") is True
        assert result.get("background") is False
        assert result.get("truncation") == "auto"

    def test_metadata_safety_identifier_user(self):
        """Metadata params pass through"""
        config = PerplexityResponsesConfig()

        params = ResponsesAPIOptionalRequestParams(
            metadata={"request_id": "req_123"},
            safety_identifier="safety_123",
            user="user_456",
            temperature=0.7,
        )

        result = config.map_openai_params(
            response_api_optional_params=params,
            model="perplexity/openai/gpt-5.2",
            drop_params=False,
        )

        assert result.get("metadata") == {"request_id": "req_123"}
        assert result.get("safety_identifier") == "safety_123"
        assert result.get("user") == "user_456"

    def test_all_supported_params_declared(self):
        """get_supported_openai_params returns complete list"""
        config = PerplexityResponsesConfig()
        supported = config.get_supported_openai_params("perplexity/openai/gpt-5.2")

        expected = [
            "max_output_tokens",
            "stream",
            "temperature",
            "top_p",
            "tools",
            "reasoning",
            "preset",
            "instructions",
            "models",
            "tool_choice",
            "parallel_tool_calls",
            "max_tool_calls",
            "text",
            "previous_response_id",
            "store",
            "background",
            "truncation",
            "metadata",
            "safety_identifier",
            "user",
            "stream_options",
            "top_logprobs",
            "prompt_cache_key",
            "frequency_penalty",
            "presence_penalty",
            "service_tier",
        ]

        for param in expected:
            assert param in supported, f"Missing supported param: {param}"

    def test_cost_transformation(self):
        """Perplexity cost dict to OpenAI float"""
        config = PerplexityResponsesConfig()

        usage_data = {
            "input_tokens": 100,
            "output_tokens": 200,
            "total_tokens": 300,
            "cost": {
                "currency": "USD",
                "input_cost": 0.0001,
                "output_cost": 0.0002,
                "total_cost": 0.0003,
            },
        }

        result = config._transform_usage(usage_data)

        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 200
        assert result["total_tokens"] == 300
        assert result["cost"] == 0.0003

    def test_cost_transformation_float_passthrough(self):
        """Cost already float passes through"""
        config = PerplexityResponsesConfig()

        usage_data = {
            "input_tokens": 100,
            "output_tokens": 200,
            "total_tokens": 300,
            "cost": 0.0005,
        }

        result = config._transform_usage(usage_data)

        assert result["cost"] == 0.0005

    def test_preset_handling(self):
        """Preset model names work"""
        config = PerplexityResponsesConfig()

        data = config.transform_responses_api_request(
            model="preset/pro-search",
            input="What is AI?",
            response_api_optional_request_params={"temperature": 0.7},
            litellm_params={},
            headers={},
        )

        assert data["preset"] == "pro-search"
        assert data["input"] == "What is AI?"
        assert "temperature" in data

    def test_get_complete_url(self):
        """Correct endpoint URL"""
        config = PerplexityResponsesConfig()

        url = config.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://api.perplexity.ai/v1/responses"

        custom_url = config.get_complete_url(
            api_base="https://custom.perplexity.ai",
            litellm_params={},
        )
        assert custom_url == "https://custom.perplexity.ai/v1/responses"

        url_with_slash = config.get_complete_url(
            api_base="https://api.perplexity.ai/",
            litellm_params={},
        )
        assert url_with_slash == "https://api.perplexity.ai/v1/responses"

    def test_perplexity_provider_config_registration(self):
        """Test that Perplexity provider returns PerplexityResponsesConfig"""
        config = ProviderConfigManager.get_provider_responses_api_config(
            model="perplexity/openai/gpt-5.2",
            provider=LlmProviders.PERPLEXITY,
        )

        assert config is not None
        assert isinstance(config, PerplexityResponsesConfig)
        assert config.custom_llm_provider == LlmProviders.PERPLEXITY
