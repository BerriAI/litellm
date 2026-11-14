import json
from pathlib import Path
from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

_DEFAULT_API_BASE: Final = "https://byesu.com/v1"


def test_byesu_is_a_registered_provider():
    assert JSONProviderRegistry.exists("byesu")
    assert "byesu" in litellm.constants.openai_compatible_providers
    assert litellm.LlmProviders.BYESU.value == "byesu"

    config: Final = JSONProviderRegistry.get("byesu")
    assert config is not None
    assert config.base_url == _DEFAULT_API_BASE
    assert config.api_key_env == "BYESU_API_KEY"
    assert config.api_base_env == "BYESU_API_BASE"


def test_byesu_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BYESU_API_KEY", "sk-test")
    monkeypatch.delenv("BYESU_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(model="byesu/gpt-6.1-sol")

    assert model == "gpt-6.1-sol"
    assert provider == "byesu"
    assert api_key == "sk-test"
    assert api_base == _DEFAULT_API_BASE


def test_byesu_explicit_api_base_and_key_win():
    config: Final = create_config_class(JSONProviderRegistry.get("byesu"))()

    api_base, api_key = config._get_openai_compatible_provider_info("https://custom.example.com/v1", "override-key")

    assert api_base == "https://custom.example.com/v1"
    assert api_key == "override-key"


def test_byesu_maps_max_completion_tokens_to_max_tokens():
    config: Final = create_config_class(JSONProviderRegistry.get("byesu"))()

    result: Final = config.map_openai_params({"max_completion_tokens": 128}, {}, "gpt-6.1-sol", False)

    assert result.get("max_tokens") == 128
    assert "max_completion_tokens" not in result


def test_byesu_declares_responses_api():
    assert JSONProviderRegistry.supports_responses_api("byesu")


def test_byesu_supported_endpoints():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    for path in (backup_path, root_path):
        endpoints = json.loads(path.read_text())["providers"]["byesu"]["endpoints"]
        assert {name for name, supported in endpoints.items() if supported} == {"chat_completions", "responses"}


def test_byesu_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"

    entry: Final = next(item for item in json.loads(fields_path.read_text()) if item["litellm_provider"] == "byesu")

    assert entry["provider"] == "BYESU"
    assert [field["key"] for field in entry["credential_fields"]] == ["api_base", "api_key"]
