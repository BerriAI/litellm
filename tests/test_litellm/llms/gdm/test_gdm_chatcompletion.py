"""
Unit tests for GDM Chat Completion configuration.

GDM (https://ai.gdm.se) provides an OpenAI-compatible API.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../.."))

import pytest

from litellm.llms.gdm.chat.transformation import GDMChatConfig


class TestGDMChatConfig:
    """Test class for GDM Chat Completion functionality"""

    @pytest.fixture
    def config(self):
        """Get GDM chat config"""
        return GDMChatConfig()

    def test_default_api_base(self, config):
        """
        Test that default API base is returned when none is provided
        """
        api_base, api_key = config._get_openai_compatible_provider_info(
            api_base=None,
            api_key="test-key"
        )

        assert api_base == "https://ai.gdm.se/api/v1"
        assert api_key == "test-key"


    def test_api_key_from_env(self, config, monkeypatch):
        """
        Test that API key is read from environment when not provided
        """
        monkeypatch.setenv("GDM_API_KEY", "env-api-key")

        api_base, api_key = config._get_openai_compatible_provider_info(
            api_base=None,
            api_key=None
        )

        assert api_key == "env-api-key"


    def test_get_supported_openai_params(self, config):
        """
        Test that get_supported_openai_params returns expected params
        (inherited from OpenAIGPTConfig)
        """
        supported_params = config.get_supported_openai_params(model="gpt-4")

        # Check common OpenAI params are supported
        assert "temperature" in supported_params
        assert "max_tokens" in supported_params
        assert "stream" in supported_params
        assert "tools" in supported_params
        assert "tool_choice" in supported_params

    def test_validate_environment(self, config):
        """
        Test that validate_environment returns proper headers
        """
        headers = config.validate_environment(
            headers={},
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key="test-api-key",
        )

        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-api-key"



    def test_map_openai_params_temperature(self, config):
        """
        Test that temperature parameter is correctly mapped
        """
        non_default_params = {"temperature": 0.7}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="gpt-4",
            drop_params=False
        )

        assert result.get("temperature") == 0.7

    def test_map_openai_params_max_tokens(self, config):
        """
        Test that max_tokens parameter is correctly mapped
        """
        non_default_params = {"max_tokens": 1000}

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="gpt-4",
            drop_params=False
        )

        assert result.get("max_tokens") == 1000

    def test_map_openai_params_multiple(self, config):
        """
        Test that multiple parameters are correctly mapped
        """
        non_default_params = {
            "temperature": 0.8,
            "max_tokens": 500,
            "top_p": 0.9,
            "presence_penalty": 0.1
        }

        result = config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
            model="gpt-4",
            drop_params=False
        )

        assert result.get("temperature") == 0.8
        assert result.get("max_tokens") == 500
        assert result.get("top_p") == 0.9
        assert result.get("presence_penalty") == 0.1
