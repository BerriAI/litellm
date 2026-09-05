"""
Unit tests for The Grid OpenAI-like provider.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
)

from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

THEGRID_BASE_URL = "https://api.thegrid.ai/v1"


def _get_config():
    provider = JSONProviderRegistry.get("thegrid")
    assert provider is not None
    config_class = create_config_class(provider)
    return config_class()


def test_thegrid_provider_registered():
    provider = JSONProviderRegistry.get("thegrid")
    assert provider is not None
    assert provider.base_url == THEGRID_BASE_URL
    assert provider.api_key_env == "THEGRID_API_KEY"
    assert provider.api_base_env == "THEGRID_API_BASE"


def test_thegrid_resolves_env_api_key(monkeypatch):
    config = _get_config()
    monkeypatch.setenv("THEGRID_API_KEY", "test-key")
    api_base, api_key = config._get_openai_compatible_provider_info(None, None)
    assert api_base == THEGRID_BASE_URL
    assert api_key == "test-key"


def test_thegrid_env_api_base_overrides_default(monkeypatch):
    config = _get_config()
    monkeypatch.setenv("THEGRID_API_KEY", "test-key")
    monkeypatch.setenv("THEGRID_API_BASE", "https://proxy.internal/v1")
    api_base, _ = config._get_openai_compatible_provider_info(None, None)
    assert api_base == "https://proxy.internal/v1"


def test_thegrid_keeps_max_completion_tokens():
    """The Grid accepts max_completion_tokens natively, so it is not remapped.

    Unlike providers that only understand the legacy max_tokens field, sending
    max_completion_tokens to /v1/chat/completions caps the response, so the
    provider entry deliberately carries no param_mappings for it.
    """
    config = _get_config()
    params = config.map_openai_params(
        non_default_params={"max_completion_tokens": 256},
        optional_params={},
        model="thegrid/text-standard",
        drop_params=False,
    )
    assert params.get("max_completion_tokens") == 256


def test_thegrid_complete_url_appends_endpoint():
    config = _get_config()
    url = config.get_complete_url(
        api_base=THEGRID_BASE_URL,
        api_key="test-key",
        model="thegrid/text-standard",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert url == f"{THEGRID_BASE_URL}/chat/completions"


def test_thegrid_supports_responses_endpoint():
    provider = JSONProviderRegistry.get("thegrid")
    assert provider is not None
    assert "/v1/chat/completions" in provider.supported_endpoints
    assert "/v1/responses" in provider.supported_endpoints
