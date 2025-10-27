import os
import sys
from unittest.mock import patch

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.nscale.chat.transformation import NscaleConfig


class TestNscaleConfig:
    def setup_method(self):
        self.config = NscaleConfig()

    def test_custom_llm_provider(self):
        """Test that custom_llm_provider returns the correct value"""
        assert self.config.custom_llm_provider == "nscale"

    def test_get_api_key(self):
        """Test that get_api_key returns the correct API key"""
        # Test with provided API key
        assert self.config.get_api_key("test-key") == "test-key"

        # Test with environment variable
        with patch(
            "litellm.llms.nscale.chat.transformation.get_secret_str",
            return_value="env-key",
        ):
            assert self.config.get_api_key() == "env-key"

        # Test with patching environment variable
        with patch.dict(os.environ, {"NSCALE_API_KEY": "env-key"}):
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
            "litellm.llms.nscale.chat.transformation.get_secret_str",
            return_value="https://env-base.com",
        ):
            assert self.config.get_api_base() == "https://env-base.com"

        # Test with default API base
        with patch(
            "litellm.llms.nscale.chat.transformation.get_secret_str", return_value=None
        ):
            assert self.config.get_api_base() == NscaleConfig.API_BASE_URL
