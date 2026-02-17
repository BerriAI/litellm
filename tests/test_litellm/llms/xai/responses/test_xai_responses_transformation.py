"""
Tests for XAI Responses API transformation

Tests the XAIResponsesAPIConfig class that handles XAI-specific
transformations for the Responses API.

Source: litellm/llms/xai/responses/transformation.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath("../../../../.."))

import pytest

from litellm.llms.xai.responses.transformation import XAIResponsesAPIConfig
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams
from litellm.types.utils import LlmProviders
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
            tools=[
                {
                    "type": "code_interpreter",
                    "container": {"type": "auto"}
                }
            ]
        )
        
        result = config.map_openai_params(
            response_api_optional_params=params,
            model="grok-4-fast",
            drop_params=False
        )
        
        assert "tools" in result
        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "code_interpreter"
        assert "container" not in result["tools"][0], "Container field should be removed"

    def test_instructions_parameter_dropped(self):
        """Test that instructions parameter is dropped for XAI"""
        config = XAIResponsesAPIConfig()
        
        params = ResponsesAPIOptionalRequestParams(
            instructions="You are a helpful assistant.",
            temperature=0.7
        )
        
        result = config.map_openai_params(
            response_api_optional_params=params,
            model="grok-4-fast",
            drop_params=False
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
        assert url == "https://api.x.ai/v1/responses", f"Expected XAI responses endpoint, got {url}"
        
        # Test with custom api_base
        custom_url = config.get_complete_url(
            api_base="https://custom.x.ai/v1", 
            litellm_params={}
        )
        assert custom_url == "https://custom.x.ai/v1/responses", f"Expected custom endpoint, got {custom_url}"
        
        # Test with trailing slash
        url_with_slash = config.get_complete_url(
            api_base="https://api.x.ai/v1/", 
            litellm_params={}
        )
        assert url_with_slash == "https://api.x.ai/v1/responses", "Should handle trailing slash"

    def test_web_search_tool_transformation(self):
        """Test that web_search tools are transformed to XAI format"""
        config = XAIResponsesAPIConfig()
        
        # Test with allowed_domains
        params = ResponsesAPIOptionalRequestParams(
            tools=[
                {
                    "type": "web_search",
                    "allowed_domains": ["wikipedia.org", "x.ai"],
                    "enable_image_understanding": True
                }
            ]
        )
        
        result = config.map_openai_params(
            response_api_optional_params=params,
            model="grok-4-1-fast",
            drop_params=False
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
                    "search_context_size": "high"  # Not supported by XAI
                }
            ]
        )
        
        result = config.map_openai_params(
            response_api_optional_params=params,
            model="grok-4-1-fast",
            drop_params=False
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
                {
                    "type": "web_search",
                    "excluded_domains": ["example.com", "test.com"]
                }
            ]
        )
        
        result = config.map_openai_params(
            response_api_optional_params=params,
            model="grok-4-1-fast",
            drop_params=False
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
                    "allowed_domains": ["d1.com", "d2.com", "d3.com", "d4.com", "d5.com", "d6.com", "d7.com"]
                }
            ]
        )
        
        result = config.map_openai_params(
            response_api_optional_params=params,
            model="grok-4-1-fast",
            drop_params=False
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
                    "enable_video_understanding": True
                }
            ]
        )
        
        result = config.map_openai_params(
            response_api_optional_params=params,
            model="grok-4-1-fast",
            drop_params=False
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
                    "excluded_x_handles": ["spam_account", "bot_account"]
                }
            ]
        )
        
        result = config.map_openai_params(
            response_api_optional_params=params,
            model="grok-4-1-fast",
            drop_params=False
        )
        
        tool = result["tools"][0]
        assert tool["excluded_x_handles"] == ["spam_account", "bot_account"]
        
    def test_mixed_tools(self):
        """Test transformation with multiple tool types"""
        config = XAIResponsesAPIConfig()
        
        params = ResponsesAPIOptionalRequestParams(
            tools=[
                {
                    "type": "code_interpreter",
                    "container": {"type": "auto"}
                },
                {
                    "type": "web_search",
                    "allowed_domains": ["wikipedia.org"]
                },
                {
                    "type": "x_search",
                    "allowed_x_handles": ["elonmusk"]
                },
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object"}
                }
            ]
        )
        
        result = config.map_openai_params(
            response_api_optional_params=params,
            model="grok-4-1-fast",
            drop_params=False
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

