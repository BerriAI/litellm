"""
Tests for XAI Responses API transformation

Tests the XAIResponsesAPIConfig class that handles XAI-specific
transformations for the Responses API.

Source: litellm/llms/xai/responses/transformation.py
"""
import sys
import os

sys.path.insert(0, os.path.abspath("../../../../.."))

import pytest
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager
from litellm.llms.xai.responses.transformation import XAIResponsesAPIConfig
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams


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

