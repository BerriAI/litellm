"""
Tests for OVHCloud Responses API transformation

Tests the OVHCloudResponsesAPIConfig class that handles OVHCloud-specific
transformations for the Responses API.

Source: litellm/llms/ovhcloud/responses/transformation.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath("../../../../.."))

import pytest

from litellm.llms.ovhcloud.responses.transformation import OVHCloudResponsesAPIConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


class TestOVHCloudResponsesAPITransformation:
    """Test OVHCloud Responses API configuration and transformations"""

    def test_ovhcloud_provider_config_registration(self):
        """Test that OVHCloud provider returns OVHCloudResponsesAPIConfig"""
        config = ProviderConfigManager.get_provider_responses_api_config(
            model="ovhcloud/gpt-oss-120b",
            provider=LlmProviders.OVHCLOUD,
        )
        
        assert config is not None, "Config should not be None for OVHCloud provider"
        assert isinstance(
            config, OVHCloudResponsesAPIConfig
        ), f"Expected OVHCloudResponsesAPIConfig, got {type(config)}"
        assert (
            config.custom_llm_provider == LlmProviders.OVHCLOUD
        ), "custom_llm_provider should be OVHCLOUD"

    def test_ovhcloud_responses_endpoint_url(self):
        """Test that get_complete_url returns correct OVHCloud endpoint"""
        config = OVHCloudResponsesAPIConfig()
        
        # Test with default OVHCloud API base
        url = config.get_complete_url(api_base=None, litellm_params={})
        assert url == "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/responses", f"Expected OVHCloud responses endpoint, got {url}"
        
        # Test with custom api_base
        custom_url = config.get_complete_url(
            api_base="https://custom.ovhcloud.example.com/v1", 
            litellm_params={}
        )
        assert custom_url == "https://custom.ovhcloud.example.com/v1/responses", f"Expected custom endpoint, got {custom_url}"
        
        # Test with trailing slash
        url_with_slash = config.get_complete_url(
            api_base="https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/", 
            litellm_params={}
        )
        assert url_with_slash == "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/responses", "Should handle trailing slash"

    def test_validate_environment_with_api_key(self):
        """Test that validate_environment sets Authorization header correctly"""
        config = OVHCloudResponsesAPIConfig()
        
        headers = {}
        litellm_params = GenericLiteLLMParams(api_key="test-api-key-123")
        
        result = config.validate_environment(
            headers=headers,
            model="ovhcloud/gpt-oss-120b",
            litellm_params=litellm_params
        )
        
        assert "Authorization" in result
        assert result["Authorization"] == "Bearer test-api-key-123"

    def test_validate_environment_missing_api_key(self):
        """Test that validate_environment raises error when API key is missing"""
        config = OVHCloudResponsesAPIConfig()
        
        headers = {}
        
        with pytest.raises(ValueError, match="OVHcloud AI Endpoints API key is required"):
            config.validate_environment(
                headers=headers,
                model="ovhcloud/gpt-oss-120b",
                litellm_params=None
            )

    def test_supported_params_includes_openai_params(self):
        """Test that get_supported_openai_params includes standard OpenAI params"""
        config = OVHCloudResponsesAPIConfig()
        supported = config.get_supported_openai_params("ovhcloud/gpt-oss-120b")
        
        # OVHCloud follows OpenAI spec, so should support standard params
        assert "model" in supported, "model should be supported"
        assert "input" in supported, "input should be supported"
        assert "temperature" in supported, "temperature should be supported"
