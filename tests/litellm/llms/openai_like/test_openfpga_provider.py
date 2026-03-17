"""
Unit tests for the OpenFPGA OpenAI-like provider.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

OPENFPGA_BASE_URL = "https://app.openfpga.ai/api/v1"


def _get_config():
    provider = JSONProviderRegistry.get("openfpga")
    assert provider is not None
    config_class = create_config_class(provider)
    return config_class()


def test_openfpga_provider_registered():
    provider = JSONProviderRegistry.get("openfpga")
    assert provider is not None
    assert provider.base_url == OPENFPGA_BASE_URL
    assert provider.api_key_env == "OPENFPGA_API_KEY"


def test_openfpga_resolves_env_api_key(monkeypatch):
    config = _get_config()
    monkeypatch.setenv("OPENFPGA_API_KEY", "ofpga_sk_live_test123")
    api_base, api_key = config._get_openai_compatible_provider_info(None, None)
    assert api_base == OPENFPGA_BASE_URL
    assert api_key == "ofpga_sk_live_test123"


def test_openfpga_complete_url_appends_endpoint():
    config = _get_config()
    url = config.get_complete_url(
        api_base=OPENFPGA_BASE_URL,
        api_key="ofpga_sk_live_test123",
        model="openfpga/llama-3.1-8b-fpga",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert url == f"{OPENFPGA_BASE_URL}/chat/completions"


def test_openfpga_provider_resolution():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, api_key, api_base = get_llm_provider(
        model="openfpga/llama-3.1-8b-fpga",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "llama-3.1-8b-fpga"
    assert provider == "openfpga"
    assert api_base == OPENFPGA_BASE_URL


def test_openfpga_provider_config_manager():
    from litellm import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_chat_config(
        model="llama-3.1-8b-fpga", provider=LlmProviders.OPENFPGA
    )

    assert config is not None
    assert config.custom_llm_provider == "openfpga"
