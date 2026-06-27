"""
Tests for XAI Responses API transformation

Tests the XAIResponsesAPIConfig class that handles XAI-specific
transformations for the Responses API.

Source: litellm/llms/xai/responses/transformation.py
"""

import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.xai.responses.transformation import XAIResponsesAPIConfig
from litellm.types.llms.openai import (
    ResponseAPIUsage,
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
)
from litellm.types.utils import LlmProviders, Usage
from litellm.utils import ProviderConfigManager


class TestXAIResponsesAPITransformation:
    """Test XAI Responses API configuration and transformations"""

    def test_xai_provider_config_registration(self):
        """Test that XAI provider returns XAIResponsesAPIConfig"""
        config = ProviderConfigManager.get_provider_responses_api_config(
            model="xai/grok-4-fast",
            provider=LlmProviders.XAI,
        )

        assert config is not None, "Config should not be None for XAI provider"
        assert isinstance(
            config, XAIResponsesAPIConfig
        ), f"Expected XAIResponsesAPIConfig, got {type(config)}"
        assert (
            config.custom_llm_provider == LlmProviders.XAI
        ), "custom_llm_provider should be XAI"

    def test_code_interpreter_container_field_removed(self):
        """Test that container field is removed from code_interpreter tools"""
        config = XAIResponsesAPIConfig()

        params = ResponsesAPIOptionalRequestParams(
            tools=[{"type": "code_interpreter", "container": {"type": "auto"}}]
        )

        result = config.map_openai_params(
            response_api_optional_params=params, model="grok-4-fast", drop_params=False
        )

        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "code_interpreter"
        assert (
            "container" not in result["tools"][0]
        ), "Container field should be removed"

    def test_instructions_parameter_dropped(self):
        """Test that instructions parameter is dropped for XAI"""
        config = XAIResponsesAPIConfig()

        params = ResponsesAPIOptionalRequestParams(
            instructions="You are a helpful assistant.", temperature=0.7
        )

        result = config.map_openai_params(
            response_api_optional_params=params, model="grok-4-fast", drop_params=False
        )

        assert "instructions" not in result, "Instructions should be dropped"
        assert result.get("temperature") == 0.7, "Other params should be preserved"

    def test_supported_params_excludes_instructions(self):
        """Test that get_supported_openai_params excludes instructions"""
        config = XAIResponsesAPIConfig()
        supported = config.get_supported_openai_params("grok-4-fast")

        assert "instructions" not in supported, "instructions should not be supported"
        assert "tools" in supported, "tools should be supported"
        assert "temperature" in supported, "temperature should be supported"
        assert "model" in supported, "model should be supported"

    def test_xai_responses_endpoint_url(self):
        """Test that get_complete_url returns correct XAI endpoint"""
        config = XAIResponsesAPIConfig()

        # Test with default XAI API base
        url = config.get_complete_url(api_base=None, litellm_params={})
        assert (
            url == "https://api.x.ai/v1/responses"
        ), f"Expected XAI responses endpoint, got {url}"

        # Test with custom api_base
        custom_url = config.get_complete_url(
            api_base="https://custom.x.ai/v1", litellm_params={}
        )
        assert (
            custom_url == "https://custom.x.ai/v1/responses"
        ), f"Expected custom endpoint, got {custom_url}"

        # Test with trailing slash
        url_with_slash = config.get_complete_url(
            api_base="https://api.x.ai/v1/", litellm_params={}
        )
        assert (
            url_with_slash == "https://api.x.ai/v1/responses"
        ), "Should handle trailing slash"

    def test_web_search_tool_transformation(self):
        """Test that web_search tools are transformed to XAI format"""
        config = XAIResponsesAPIConfig()

        # Test with allowed_domains
        params = ResponsesAPIOptionalRequestParams(
            tools=[
                {
                    "type": "web_search",
                    "allowed_domains": ["wikipedia.org", "x.ai"],
                    "enable_image_understanding": True,
                }
            ]
        )

        result = config.map_openai_params(
            response_api_optional_params=params,
            model="grok-4-1-fast",
            drop_params=False,
        )

        assert "tools" in result
        assert len(result["tools"]) == 1
        tool = result["tools"][0]
        assert tool["type"] == "web_search"
        assert "filters" in tool
        assert tool["filters"]["allowed_domains"] == ["wikipedia.org", "x.ai"]
        assert tool["enable_image_understanding"] is True

    def test_web_search_search_context_size_removed(self):
        """Test that search_context_size is removed from web_search tools"""
        config = XAIResponsesAPIConfig()

        params = ResponsesAPIOptionalRequestParams(
            tools=[
                {
                    "type": "web_search",
                    "search_context_size": "high",  # Not supported by XAI
                }
            ]
        )

        result = config.map_openai_params(
            response_api_optional_params=params,
            model="grok-4-1-fast",
            drop_params=False,
        )

        assert "tools" in result
        assert len(result["tools"]) == 1
        tool = result["tools"][0]
        assert tool["type"] == "web_search"
        assert "search_context_size" not in tool

    def test_web_search_excluded_domains(self):
        """Test web_search with excluded_domains"""
        config = XAIResponsesAPIConfig()

        params = ResponsesAPIOptionalRequestParams(
            tools=[
                {"type": "web_search", "excluded_domains": ["example.com", "test.com"]}
            ]
        )

        result = config.map_openai_params(
            response_api_optional_params=params,
            model="grok-4-1-fast",
            drop_params=False,
        )

        tool = result["tools"][0]
        assert "filters" in tool
        assert tool["filters"]["excluded_domains"] == ["example.com", "test.com"]

    def test_web_search_domains_limit(self):
        """Test that allowed_domains and excluded_domains are limited to 5"""
        config = XAIResponsesAPIConfig()

        # Test with more than 5 allowed_domains
        params = ResponsesAPIOptionalRequestParams(
            tools=[
                {
                    "type": "web_search",
                    "allowed_domains": [
                        "d1.com",
                        "d2.com",
                        "d3.com",
                        "d4.com",
                        "d5.com",
                        "d6.com",
                        "d7.com",
                    ],
                }
            ]
        )

        result = config.map_openai_params(
            response_api_optional_params=params,
            model="grok-4-1-fast",
            drop_params=False,
        )

        tool = result["tools"][0]
        assert len(tool["filters"]["allowed_domains"]) == 7

    def test_x_search_tool_transformation(self):
        """Test that x_search tools are transformed correctly"""
        config = XAIResponsesAPIConfig()

        params = ResponsesAPIOptionalRequestParams(
            tools=[
                {
                    "type": "x_search",
                    "allowed_x_handles": ["elonmusk", "xai"],
                    "from_date": "2025-01-01",
                    "to_date": "2025-01-28",
                    "enable_image_understanding": True,
                    "enable_video_understanding": True,
                }
            ]
        )

        result = config.map_openai_params(
            response_api_optional_params=params,
            model="grok-4-1-fast",
            drop_params=False,
        )

        assert "tools" in result
        assert len(result["tools"]) == 1
        tool = result["tools"][0]
        assert tool["type"] == "x_search"
        assert tool["allowed_x_handles"] == ["elonmusk", "xai"]
        assert tool["from_date"] == "2025-01-01"
        assert tool["to_date"] == "2025-01-28"
        assert tool["enable_image_understanding"] is True
        assert tool["enable_video_understanding"] is True

    def test_x_search_excluded_handles(self):
        """Test x_search with excluded_x_handles"""
        config = XAIResponsesAPIConfig()

        params = ResponsesAPIOptionalRequestParams(
            tools=[
                {
                    "type": "x_search",
                    "excluded_x_handles": ["spam_account", "bot_account"],
                }
            ]
        )

        result = config.map_openai_params(
            response_api_optional_params=params,
            model="grok-4-1-fast",
            drop_params=False,
        )

        tool = result["tools"][0]
        assert tool["excluded_x_handles"] == ["spam_account", "bot_account"]

    def test_mixed_tools(self):
        """Test transformation with multiple tool types"""
        config = XAIResponsesAPIConfig()

        params = ResponsesAPIOptionalRequestParams(
            tools=[
                {"type": "code_interpreter", "container": {"type": "auto"}},
                {"type": "web_search", "allowed_domains": ["wikipedia.org"]},
                {"type": "x_search", "allowed_x_handles": ["elonmusk"]},
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"},
                },
            ]
        )

        result = config.map_openai_params(
            response_api_optional_params=params,
            model="grok-4-1-fast",
            drop_params=False,
        )

        assert len(result["tools"]) == 4

        # Verify code_interpreter
        assert result["tools"][0]["type"] == "code_interpreter"
        assert "container" not in result["tools"][0]

        # Verify web_search
        assert result["tools"][1]["type"] == "web_search"
        assert "filters" in result["tools"][1]

        # Verify x_search
        assert result["tools"][2]["type"] == "x_search"
        assert result["tools"][2]["allowed_x_handles"] == ["elonmusk"]

        # Verify function tool is unchanged
        assert result["tools"][3]["type"] == "function"
        assert result["tools"][3]["name"] == "get_weather"


class TestXAIResponsesToolUsageAttach:
    """Tests for server_side_tool_usage_details attach helpers (cost billing)."""

    _TOOL_DETAILS = {
        "web_search_calls": 2,
        "x_search_calls": 0,
        "code_interpreter_calls": 0,
        "file_search_calls": 0,
        "mcp_calls": 0,
        "document_search_calls": 0,
    }

    def test_server_side_tool_usage_details_from_usage_dict(self):
        details = XAIResponsesAPIConfig._server_side_tool_usage_details_from_usage(
            {"server_side_tool_usage_details": self._TOOL_DETAILS}
        )
        assert details == self._TOOL_DETAILS

    def test_server_side_tool_usage_details_from_usage_attr(self):
        usage = Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        setattr(usage, "server_side_tool_usage_details", self._TOOL_DETAILS)
        details = XAIResponsesAPIConfig._server_side_tool_usage_details_from_usage(
            usage
        )
        assert details == self._TOOL_DETAILS

    def test_server_side_tool_usage_details_from_model_extra(self):
        usage = ResponseAPIUsage(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
            server_side_tool_usage_details=self._TOOL_DETAILS,
        )
        details = XAIResponsesAPIConfig._server_side_tool_usage_details_from_usage(
            usage
        )
        assert details == self._TOOL_DETAILS

    def test_server_side_tool_usage_details_from_usage_none(self):
        assert (
            XAIResponsesAPIConfig._server_side_tool_usage_details_from_usage(None)
            is None
        )
        assert (
            XAIResponsesAPIConfig._server_side_tool_usage_details_from_usage(
                Usage(prompt_tokens=1, completion_tokens=0, total_tokens=1)
            )
            is None
        )

    def test_attach_noop_when_usage_missing(self):
        response = ResponsesAPIResponse.model_construct(
            id="resp_1", created_at=0, output=[], usage=None
        )
        XAIResponsesAPIConfig._attach_server_side_tool_usage_details_to_usage(response)
        assert response.usage is None

    def test_attach_noop_when_details_missing(self):
        usage = ResponseAPIUsage(input_tokens=3, output_tokens=1, total_tokens=4)
        response = ResponsesAPIResponse.model_construct(
            id="resp_2", created_at=0, output=[], usage=usage
        )
        XAIResponsesAPIConfig._attach_server_side_tool_usage_details_to_usage(response)
        assert isinstance(response.usage, ResponseAPIUsage)

    def test_attach_converts_response_api_usage_to_chat_usage(self):
        usage = ResponseAPIUsage(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            server_side_tool_usage_details=self._TOOL_DETAILS,
        )
        response = ResponsesAPIResponse.model_construct(
            id="resp_3", created_at=0, output=[], usage=usage
        )
        XAIResponsesAPIConfig._attach_server_side_tool_usage_details_to_usage(response)

        assert isinstance(response.usage, Usage)
        assert response.usage.prompt_tokens == 100
        assert response.usage.completion_tokens == 20
        assert getattr(response.usage, "server_side_tool_usage_details") == (
            self._TOOL_DETAILS
        )
        assert response.usage.prompt_tokens_details is not None
        assert response.usage.prompt_tokens_details.web_search_requests == 2

    def test_attach_updates_existing_chat_usage_in_place(self):
        usage = Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10)
        setattr(usage, "server_side_tool_usage_details", self._TOOL_DETAILS)
        response = ResponsesAPIResponse.model_construct(
            id="resp_4", created_at=0, output=[], usage=usage
        )
        XAIResponsesAPIConfig._attach_server_side_tool_usage_details_to_usage(response)

        assert response.usage is usage
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.web_search_requests == 2

    def test_transform_streaming_response_completed_attaches_tool_usage(self):
        config = XAIResponsesAPIConfig()
        chunk = {
            "type": "response.completed",
            "response": {
                "id": "resp_stream",
                "created_at": 1,
                "output": [],
                "usage": {
                    "input_tokens": 50,
                    "output_tokens": 10,
                    "total_tokens": 60,
                    "server_side_tool_usage_details": self._TOOL_DETAILS,
                },
            },
        }
        event = config.transform_streaming_response(
            model="grok-4.3", parsed_chunk=chunk, logging_obj=MagicMock()
        )

        assert isinstance(event, ResponseCompletedEvent)
        assert isinstance(event.response.usage, Usage)
        assert getattr(event.response.usage, "server_side_tool_usage_details") == (
            self._TOOL_DETAILS
        )
        assert event.response.usage.prompt_tokens_details is not None
        assert event.response.usage.prompt_tokens_details.web_search_requests == 2

    def test_transform_streaming_response_non_terminal_event_unchanged(self):
        config = XAIResponsesAPIConfig()
        chunk = {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "hi",
        }
        event = config.transform_streaming_response(
            model="grok-4.3", parsed_chunk=chunk, logging_obj=MagicMock()
        )
        assert getattr(event, "type", None) is not None
        assert not isinstance(
            event,
            (ResponseCompletedEvent, ResponseIncompleteEvent, ResponseFailedEvent),
        )
