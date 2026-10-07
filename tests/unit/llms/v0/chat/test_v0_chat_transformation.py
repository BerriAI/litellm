import os
from unittest import mock

import litellm

from litellm.llms.v0.chat.transformation import V0ChatConfig


def test_v0_config_initialization():
    """Test V0ChatConfig initializes correctly"""
    config = V0ChatConfig()
    assert config.custom_llm_provider == "v0"


def test_v0_get_openai_compatible_provider_info():
    """Test v0 provider info retrieval"""
    config = V0ChatConfig()
    with mock.patch.dict(os.environ, {}, clear=True):
        api_base, api_key = config.get_openai_compatible_provider_info(None, None)
        assert api_base == "https://api.v0.dev/v1"
        assert api_key is None
    with mock.patch.dict(os.environ, {"V0_API_KEY": "test-key", "V0_API_BASE": "https://custom.v0.ai/v1"}):
        api_base, api_key = config.get_openai_compatible_provider_info(None, None)
        assert api_base == "https://custom.v0.ai/v1"
        assert api_key == "test-key"
    with mock.patch.dict(os.environ, {"V0_API_KEY": "env-key", "V0_API_BASE": "https://env.v0.ai/v1"}):
        api_base, api_key = config.get_openai_compatible_provider_info("https://param.v0.ai/v1", "param-key")
        assert api_base == "https://param.v0.ai/v1"
        assert api_key == "param-key"


def test_get_llm_provider_v0():
    """Test that get_llm_provider correctly identifies v0"""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, api_key, api_base = get_llm_provider("v0/gpt-4-turbo")
    assert model == "gpt-4-turbo"
    assert provider == "v0"
    model, provider, api_key, api_base = get_llm_provider("gpt-4-turbo", api_base="https://api.v0.dev/v1")
    assert model == "gpt-4-turbo"
    assert provider == "v0"
    assert api_base == "https://api.v0.dev/v1"


def test_v0_in_provider_lists():
    """Test that v0 is registered in all necessary provider lists"""
    assert "v0" in litellm.openai_compatible_providers
    assert "v0" in litellm.provider_list
    assert "https://api.v0.dev/v1" in litellm.openai_compatible_endpoints


def test_v0_supported_params():
    """Test that v0 returns only the supported parameters"""
    config = V0ChatConfig()
    supported_params = config.get_supported_openai_params("v0/v0-1.5-md")
    expected_params = ["messages", "model", "stream", "tools", "tool_choice"]
    assert set(supported_params) == set(expected_params)
