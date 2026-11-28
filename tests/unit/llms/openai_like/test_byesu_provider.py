import json
from pathlib import Path
from typing import Final

import pytest

import litellm
from litellm.exceptions import AuthenticationError
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai_like.common_utils import missing_json_provider_key_error
from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

_DEFAULT_API_BASE: Final = "https://byesu.com/v1"


def test_byesu_is_a_registered_provider():
    assert JSONProviderRegistry.exists("byesu")
    assert litellm.LlmProviders.BYESU.value == "byesu"
    assert "byesu" in litellm.provider_list
    assert "byesu" in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints

    config: Final = JSONProviderRegistry.get("byesu")
    assert config is not None
    assert config.base_url == _DEFAULT_API_BASE
    assert config.api_key_env == "BYESU_API_KEY"
    assert config.api_base_env == "BYESU_API_BASE"


def test_byesu_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BYESU_API_KEY", "sk-byesu-test")
    monkeypatch.delenv("BYESU_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="byesu/gpt-6.2-sol",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "gpt-6.2-sol"
    assert provider == "byesu"
    assert api_key == "sk-byesu-test"
    assert api_base == _DEFAULT_API_BASE


def test_byesu_explicit_api_base_and_key_win(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BYESU_API_KEY", "sk-env-key")
    monkeypatch.setenv("BYESU_API_BASE", "https://env.example.com/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="byesu/gpt-6.2-sol",
        custom_llm_provider=None,
        api_base="https://custom.example.com/v1",
        api_key="override-key",
    )

    assert provider == "byesu"
    assert api_base == "https://custom.example.com/v1"
    assert api_key == "override-key"


def test_byesu_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BYESU_API_KEY", "sk-byesu-env")
    monkeypatch.delenv("BYESU_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="gpt-6.2-sol",
        custom_llm_provider=None,
        api_base=_DEFAULT_API_BASE,
        api_key=None,
    )

    assert model == "gpt-6.2-sol"
    assert provider == "byesu"
    assert api_key == "sk-byesu-env"
    assert api_base == _DEFAULT_API_BASE


def test_byesu_missing_key_does_not_fall_back_to_openai_secret(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("BYESU_API_KEY", raising=False)
    monkeypatch.delenv("BYESU_API_BASE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-secret")
    monkeypatch.setattr(litellm, "api_key", None)

    _, provider, api_key, _ = get_llm_provider(
        model="byesu/gpt-6.2-sol",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )
    error: Final = missing_json_provider_key_error(
        custom_llm_provider=provider,
        model="gpt-6.2-sol",
        api_key=api_key,
        uses_openai_client=False,
    )

    assert provider == "byesu"
    assert api_key is None
    assert isinstance(error, AuthenticationError)
    assert "BYESU_API_KEY" in str(error)

    with pytest.raises(AuthenticationError, match="BYESU_API_KEY"):
        litellm.completion(
            model="byesu/gpt-6.2-sol",
            messages=[{"role": "user", "content": "hi"}],
        )


def test_byesu_preserves_max_completion_tokens_and_reasoning_params():
    provider_cfg: Final = JSONProviderRegistry.get("byesu")
    assert provider_cfg is not None
    config: Final = create_config_class(provider_cfg)()

    mapped: Final = config.map_openai_params(
        {"max_completion_tokens": 256, "reasoning_effort": "medium"},
        {},
        "gpt-6.2-sol",
        False,
    )

    assert mapped.get("max_completion_tokens") == 256
    assert mapped.get("reasoning_effort") == "medium"
    assert "max_tokens" not in mapped


def test_byesu_declares_responses_api():
    assert JSONProviderRegistry.supports_responses_api("byesu")


def test_byesu_provider_endpoints_support_mirrors_backup():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    backup: Final = json.loads(backup_path.read_text())["providers"]["byesu"]
    root: Final = json.loads(root_path.read_text())["providers"]["byesu"]

    assert backup == root
    assert root["display_name"] == "Byesu (`byesu`)"


def test_byesu_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    entry: Final = next(item for item in json.loads(fields_path.read_text()) if item["litellm_provider"] == "byesu")

    assert entry["provider"] == "BYESU"
    assert entry["provider_display_name"] == "Byesu"
    assert entry["default_model_placeholder"] == "byesu/gpt-6.2-sol"
    assert {field["key"]: field["required"] for field in entry["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }
