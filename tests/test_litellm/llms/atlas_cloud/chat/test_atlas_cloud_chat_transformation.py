import os
import sys
from unittest.mock import patch

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.llms.atlas_cloud.chat.transformation import AtlasCloudConfig


class TestAtlasCloudConfig:
    def setup_method(self):
        self.config = AtlasCloudConfig()

    def test_custom_llm_provider(self):
        """Test that custom_llm_provider returns the correct value"""
        assert self.config.custom_llm_provider == "atlas_cloud"

    def test_get_api_key(self):
        """Test that get_api_key returns the correct API key"""
        # Test with provided API key
        assert self.config.get_api_key("test-key") == "test-key"

        # Test with environment variable
        with patch(
            "litellm.llms.atlas_cloud.chat.transformation.get_secret_str",
            return_value="env-key",
        ):
            assert self.config.get_api_key() == "env-key"

        # Test with patching environment variable
        with patch.dict(os.environ, {"ATLASCLOUD_API_KEY": "env-key"}):
            assert self.config.get_api_key() == "env-key"

    def test_get_api_base(self):
        """Test that get_api_base returns the correct API base URL"""
        # Test with provided API base
        assert (
            self.config.get_api_base("https://custom-base.com")
            == "https://custom-base.com"
        )

        # Test with environment variable
        with patch(
            "litellm.llms.atlas_cloud.chat.transformation.get_secret_str",
            return_value="https://env-base.com",
        ):
            assert self.config.get_api_base() == "https://env-base.com"

        # Test with default API base
        with patch(
            "litellm.llms.atlas_cloud.chat.transformation.get_secret_str",
            return_value=None,
        ):
            assert self.config.get_api_base() == AtlasCloudConfig.API_BASE_URL

    def test_supported_openai_params(self):
        """Atlas Cloud is OpenAI-compatible and supports tools / streaming."""
        params = self.config.get_supported_openai_params(
            model="deepseek-ai/deepseek-v4-pro"
        )
        for expected in ("max_tokens", "temperature", "stream", "tools", "tool_choice"):
            assert expected in params


class TestAtlasCloudProviderResolution:
    def test_get_llm_provider_prefix(self):
        """`atlas_cloud/<model>` resolves to the atlas_cloud provider and default base."""
        model, provider, _dynamic_key, api_base = litellm.get_llm_provider(
            model="atlas_cloud/deepseek-ai/deepseek-v4-pro"
        )
        assert provider == "atlas_cloud"
        assert model == "deepseek-ai/deepseek-v4-pro"
        assert api_base == AtlasCloudConfig.API_BASE_URL

    def test_provider_registered(self):
        """atlas_cloud is registered in the provider list and provider enum."""
        from litellm.types.utils import LlmProviders

        assert "atlas_cloud" in litellm.provider_list
        assert LlmProviders.ATLAS_CLOUD.value == "atlas_cloud"
