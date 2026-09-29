"""
Unit tests for the DemonRoute OpenAI-like provider (JSON-configured).
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

DEMONROUTE_BASE_URL = "https://api.demonroute.com/v1"


def _get_config():
    provider = JSONProviderRegistry.get("demonroute")
    assert provider is not None
    config_class = create_config_class(provider)
    return config_class()


def test_demonroute_provider_registered():
    provider = JSONProviderRegistry.get("demonroute")
    assert provider is not None
    assert provider.base_url == DEMONROUTE_BASE_URL
    assert provider.api_key_env == "DEMONROUTE_API_KEY"
    assert provider.api_base_env == "DEMONROUTE_API_BASE"


def test_demonroute_resolves_env_api_key(monkeypatch):
    config = _get_config()
    monkeypatch.delenv("DEMONROUTE_API_BASE", raising=False)
    monkeypatch.setenv("DEMONROUTE_API_KEY", "test-key")
    api_base, api_key = config._get_openai_compatible_provider_info(None, None)
    assert api_base == DEMONROUTE_BASE_URL
    assert api_key == "test-key"


def test_demonroute_complete_url_appends_endpoint():
    config = _get_config()
    url = config.get_complete_url(
        api_base=DEMONROUTE_BASE_URL,
        api_key="test-key",
        model="demonroute/NousResearch/Hermes-4-70B",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert url == f"{DEMONROUTE_BASE_URL}/chat/completions"


def test_demonroute_model_id_keeps_org_segment():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, _, api_base = get_llm_provider(
        model="demonroute/NousResearch/Hermes-4-70B",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )
    assert provider == "demonroute"
    assert model == "NousResearch/Hermes-4-70B"
    assert api_base == DEMONROUTE_BASE_URL
