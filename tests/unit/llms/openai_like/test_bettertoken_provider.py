import json
from pathlib import Path

import pytest

import litellm
from litellm.llms.openai_like.dynamic_config import create_responses_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry, SimpleProviderConfig
from litellm.llms.openai_like.messages.transformation import JSONProviderAnthropicMessagesConfig
from litellm.types.router import GenericLiteLLMParams

_OPENAI_SENTINEL = "sk-openai-sentinel"
_GLOBAL_SENTINEL = "sk-litellm-global"
_OPENAI_ATTR_SENTINEL = "sk-openai-attr"


def _provider() -> SimpleProviderConfig:
    provider = JSONProviderRegistry.get("bettertoken")
    assert provider is not None
    return provider


def _install_openai_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BETTERTOKEN_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", _OPENAI_SENTINEL)
    monkeypatch.setenv("OPENAI_ORGANIZATION", "org-openai-sentinel")
    monkeypatch.setattr(litellm, "api_key", _GLOBAL_SENTINEL)
    monkeypatch.setattr(litellm, "openai_key", _OPENAI_ATTR_SENTINEL)
    monkeypatch.setattr(litellm, "organization", "org-litellm-global")


def _assert_openai_secrets_absent(text: str) -> None:
    assert _OPENAI_SENTINEL not in text
    assert _GLOBAL_SENTINEL not in text
    assert _OPENAI_ATTR_SENTINEL not in text
    assert "org-openai-sentinel" not in text


def test_bettertoken_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("BETTERTOKEN_API_KEY", "bt-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="bettertoken/claude-sonnet-5-5",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "claude-sonnet-5-5"
    assert provider == "bettertoken"
    assert api_key == "bt-test-key"
    assert api_base == "https://bettertoken.ai/v1"


def test_bettertoken_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("BETTERTOKEN_API_KEY", "bt-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="bettertoken/claude-sonnet-5-5",
        custom_llm_provider=None,
        api_base="https://bettertoken.internal.example/v1",
        api_key="bt-explicit-key",
    )

    assert provider == "bettertoken"
    assert api_key == "bt-explicit-key"
    assert api_base == "https://bettertoken.internal.example/v1"


def test_bettertoken_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    bt = next(p for p in providers if p["litellm_provider"] == "bettertoken")

    assert bt["provider_display_name"] == "BetterToken"
    assert bt["default_model_placeholder"] == "bettertoken/claude-sonnet-5-5"
    assert {f["key"]: f["required"] for f in bt["credential_fields"]} == {"api_base": False, "api_key": True}


def test_bettertoken_cost_map_entries_exist():
    package = Path(litellm.__file__).resolve().parent
    root = json.loads((package.parent / "model_prices_and_context_window.json").read_text())
    backup = json.loads((package / "model_prices_and_context_window_backup.json").read_text())
    root_entries = {key: value for key, value in root.items() if key.startswith("bettertoken/")}
    backup_entries = {key: value for key, value in backup.items() if key.startswith("bettertoken/")}
    assert root_entries
    assert backup_entries == root_entries
    entry = litellm.model_cost["bettertoken/claude-sonnet-5-5"]
    assert entry == root_entries["bettertoken/claude-sonnet-5-5"]
    assert entry["litellm_provider"] == "bettertoken"
    assert entry["input_cost_per_token"] > 0
    assert entry["output_cost_per_token"] > 0


def test_unset_bettertoken_key_is_not_replaced_with_openai_credentials(monkeypatch: pytest.MonkeyPatch):
    _install_openai_credentials(monkeypatch)

    with pytest.raises(litellm.AuthenticationError) as caught:
        litellm.completion(
            model="bettertoken/claude-sonnet-5-5",
            messages=[{"role": "user", "content": "hi"}],
        )

    assert caught.value.llm_provider == "bettertoken"
    assert caught.value.model == "claude-sonnet-5-5"
    _assert_openai_secrets_absent(str(caught.value))


def test_bettertoken_messages_do_not_send_a_global_key(monkeypatch: pytest.MonkeyPatch):
    _install_openai_credentials(monkeypatch)
    config = JSONProviderAnthropicMessagesConfig(_provider())

    with pytest.raises(litellm.AuthenticationError) as caught:
        config.validate_anthropic_messages_environment(
            headers={},
            model="claude-sonnet-5-5",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
        )

    assert caught.value.llm_provider == "bettertoken"
    _assert_openai_secrets_absent(str(caught.value))

    monkeypatch.setenv("BETTERTOKEN_API_KEY", "bt-env-key")
    headers, _api_base = config.validate_anthropic_messages_environment(
        headers={},
        model="claude-sonnet-5-5",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params={},
    )
    assert headers["authorization"] == "Bearer bt-env-key"
    _assert_openai_secrets_absent(json.dumps(headers))


def test_bettertoken_responses_do_not_send_openai_credentials(monkeypatch: pytest.MonkeyPatch):
    _install_openai_credentials(monkeypatch)
    config = create_responses_config_class(_provider())()

    missing = config.validate_environment(
        headers={},
        model="claude-sonnet-5-5",
        litellm_params=GenericLiteLLMParams(),
    )
    assert "Authorization" not in missing
    _assert_openai_secrets_absent(json.dumps(missing))

    monkeypatch.setenv("BETTERTOKEN_API_KEY", "bt-env-key")
    present = config.validate_environment(
        headers={},
        model="claude-sonnet-5-5",
        litellm_params=GenericLiteLLMParams(),
    )
    assert present["Authorization"] == "Bearer bt-env-key"
    _assert_openai_secrets_absent(json.dumps({key: value for key, value in present.items() if key != "Authorization"}))
