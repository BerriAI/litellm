"""
Tests for Sail Research Responses API transformation

Tests the SailResearchResponsesConfig class that handles Sail-specific
transformations for the Responses API.

Source: litellm/llms/sail_research/responses/transformation.py
"""

import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.sail_research.responses.transformation import (
    SailResearchResponsesConfig,
)
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


class TestSailResearchResponsesTransformation:
    """Test Sail Research Responses API configuration and transformations"""

    def test_supported_params(self):
        """get_supported_openai_params returns Sail-specific restricted list"""
        config = SailResearchResponsesConfig()
        supported = config.get_supported_openai_params(
            "sail_research/deepseek-ai/DeepSeek-V3.2"
        )

        expected = [
            "model",
            "input",
            "temperature",
            "top_p",
            "max_output_tokens",
            "tools",
            "tool_choice",
            "text",
            "reasoning",
            "metadata",
            "background",
        ]

        for param in expected:
            assert param in supported, f"Missing supported param: {param}"

        # Params that Sail does NOT support
        unsupported = [
            "instructions",
            "previous_response_id",
            "parallel_tool_calls",
            "store",
            "truncation",
        ]

        for param in unsupported:
            assert param not in supported, f"Param should not be supported: {param}"

    def test_should_fake_stream(self):
        """Sail does not support streaming — should_fake_stream must return True"""
        config = SailResearchResponsesConfig()
        assert config.should_fake_stream() is True
        assert (
            config.should_fake_stream(model="deepseek-ai/DeepSeek-V3.2", stream=True)
            is True
        )

    def test_custom_llm_provider(self):
        """Provider enum is SAIL_RESEARCH"""
        config = SailResearchResponsesConfig()
        assert config.custom_llm_provider == LlmProviders.SAIL_RESEARCH

    def test_get_complete_url_default(self):
        """Default URL points to Sail Research API"""
        config = SailResearchResponsesConfig()
        url = config.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://api.sailresearch.com/v1/responses"

    def test_get_complete_url_custom(self):
        """Custom api_base is respected"""
        config = SailResearchResponsesConfig()
        url = config.get_complete_url(
            api_base="https://custom.sail.dev", litellm_params={}
        )
        assert url == "https://custom.sail.dev/v1/responses"

    def test_get_complete_url_trailing_slash(self):
        """Trailing slash in api_base is handled"""
        config = SailResearchResponsesConfig()
        url = config.get_complete_url(
            api_base="https://api.sailresearch.com/", litellm_params={}
        )
        assert url == "https://api.sailresearch.com/v1/responses"

    def test_no_double_v1_in_default_url(self):
        """Regression: default api_base must not produce /v1/v1/responses"""
        config = SailResearchResponsesConfig()
        # Simulate the default base URL that get_llm_provider_logic resolves
        url = config.get_complete_url(
            api_base="https://api.sailresearch.com", litellm_params={}
        )
        assert url == "https://api.sailresearch.com/v1/responses"
        assert "/v1/v1/" not in url

    def test_stream_stripped_from_request(self):
        """stream param is removed from the request body"""
        config = SailResearchResponsesConfig()

        data = config.transform_responses_api_request(
            model="deepseek-ai/DeepSeek-V3.2",
            input="Hello",
            response_api_optional_request_params={"stream": True, "temperature": 0.7},
            litellm_params={},
            headers={},
        )

        assert "stream" not in data
        assert data["temperature"] == 0.7

    def test_json_object_format_stripped(self):
        """text.format.type json_object is stripped (Sail only accepts json_schema)"""
        config = SailResearchResponsesConfig()

        data = config.transform_responses_api_request(
            model="deepseek-ai/DeepSeek-V3.2",
            input="Return JSON",
            response_api_optional_request_params={
                "text": {
                    "format": {"type": "json_object"},
                },
            },
            litellm_params={},
            headers={},
        )

        # The json_object format should be stripped
        text_config = data.get("text", {})
        assert "format" not in text_config

    def test_json_schema_format_preserved(self):
        """text.format.type json_schema is preserved"""
        config = SailResearchResponsesConfig()

        schema_format = {
            "type": "json_schema",
            "name": "my_schema",
            "schema": {"type": "object", "properties": {"x": {"type": "number"}}},
        }

        data = config.transform_responses_api_request(
            model="deepseek-ai/DeepSeek-V3.2",
            input="Return JSON",
            response_api_optional_request_params={
                "text": {"format": schema_format},
            },
            litellm_params={},
            headers={},
        )

        assert data["text"]["format"]["type"] == "json_schema"

    def test_function_tool_passthrough(self):
        """Function tools are preserved"""
        config = SailResearchResponsesConfig()

        params = ResponsesAPIOptionalRequestParams(
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        )

        result = config.map_openai_params(
            response_api_optional_params=params,
            model="sail_research/deepseek-ai/DeepSeek-V3.2",
            drop_params=False,
        )

        assert "tools" in result
        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["function"]["name"] == "get_weather"

    def test_background_param_supported(self):
        """background: true is a supported Sail-specific param"""
        config = SailResearchResponsesConfig()

        params = ResponsesAPIOptionalRequestParams(background=True)

        result = config.map_openai_params(
            response_api_optional_params=params,
            model="sail_research/deepseek-ai/DeepSeek-V3.2",
            drop_params=False,
        )

        assert result.get("background") is True

    def test_provider_config_registration(self):
        """ProviderConfigManager returns SailResearchResponsesConfig"""
        config = ProviderConfigManager.get_provider_responses_api_config(
            model="sail_research/deepseek-ai/DeepSeek-V3.2",
            provider=LlmProviders.SAIL_RESEARCH,
        )

        assert config is not None
        assert isinstance(config, SailResearchResponsesConfig)
        assert config.custom_llm_provider == LlmProviders.SAIL_RESEARCH

    def test_no_native_websocket(self):
        """Sail does not support native WebSocket"""
        config = SailResearchResponsesConfig()
        assert config.supports_native_websocket() is False

    def test_validate_environment_sets_auth_header(self):
        """API key is set in Authorization header"""
        config = SailResearchResponsesConfig()
        from litellm.types.router import GenericLiteLLMParams

        headers = config.validate_environment(
            headers={},
            model="deepseek-ai/DeepSeek-V3.2",
            litellm_params=GenericLiteLLMParams(api_key="sk-test-123"),
        )

        assert headers["Authorization"] == "Bearer sk-test-123"

    def test_successful_response_passes_through(self):
        """Normal completed response delegates to base OpenAI handler"""
        from litellm.litellm_core_utils.litellm_logging import (
            Logging as LiteLLMLoggingObj,
        )

        config = SailResearchResponsesConfig()

        success_body = {
            "id": "resp_123",
            "object": "response",
            "created_at": 1700000000,
            "status": "completed",
            "model": "deepseek-ai/DeepSeek-V3.2",
            "output": [
                {
                    "type": "message",
                    "id": "msg_123",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "Hello!", "annotations": []}
                    ],
                }
            ],
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
        }

        raw_response = httpx.Response(
            status_code=200,
            json=success_body,
            request=httpx.Request("POST", "https://api.sailresearch.com/v1/responses"),
        )

        logging_obj = LiteLLMLoggingObj(
            model="sail_research/deepseek-ai/DeepSeek-V3.2",
            messages=[],
            stream=False,
            call_type="responses",
            start_time=None,
            litellm_call_id="test",
            function_id="test",
        )

        response = config.transform_response_api_response(
            model="sail_research/deepseek-ai/DeepSeek-V3.2",
            raw_response=raw_response,
            logging_obj=logging_obj,
        )

        assert response.id == "resp_123"
        assert response.status == "completed"
