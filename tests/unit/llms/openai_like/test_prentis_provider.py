"""
Tests for Prentis provider configuration and integration.
"""

import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "https://compute.prentis.ai/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_MODEL: Final = "prentis/qwen3.8-27b"
PRENTIS_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("prentis/")))


def test_prentis_is_a_registered_provider():
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    assert litellm.LlmProviders.PRENTIS.value == "prentis"
    assert "prentis" in litellm.provider_list
    assert "prentis" in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints

    prentis = JSONProviderRegistry.get("prentis")
    assert prentis is not None
    assert prentis.base_url == _DEFAULT_API_BASE
    assert prentis.api_key_env == "PRENTIS_API_KEY"
    assert prentis.api_base_env == "PRENTIS_API_BASE"


def test_prentis_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PRENTIS_API_KEY", "prentis-test-key")
    monkeypatch.delenv("PRENTIS_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model=_MODEL, custom_llm_provider=None, api_base=None, api_key=None
    )

    assert model == "qwen3.8-27b"
    assert provider == "prentis"
    assert api_key == "prentis-test-key"
    assert api_base == _DEFAULT_API_BASE


def test_prentis_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PRENTIS_API_KEY", "prentis-test-key")
    monkeypatch.setenv("PRENTIS_API_BASE", "https://prentis.env.example/v1")

    _, provider, _, api_base = get_llm_provider(model=_MODEL, custom_llm_provider=None, api_base=None, api_key=None)

    assert provider == "prentis"
    assert api_base == "https://prentis.env.example/v1"


def test_prentis_explicit_credentials_win(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PRENTIS_API_KEY", "prentis-env-key")
    monkeypatch.setenv("PRENTIS_API_BASE", "https://prentis.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model=_MODEL,
        custom_llm_provider=None,
        api_base="https://prentis.internal.example/v1",
        api_key="prentis-explicit-key",
    )

    assert provider == "prentis"
    assert api_key == "prentis-explicit-key"
    assert api_base == "https://prentis.internal.example/v1"


def test_prentis_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PRENTIS_API_KEY", "prentis-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="qwen3.8-27b", custom_llm_provider=None, api_base=_DEFAULT_API_BASE, api_key=None
    )

    assert model == "qwen3.8-27b"
    assert provider == "prentis"
    assert api_key == "prentis-env-key"
    assert api_base == _DEFAULT_API_BASE


@pytest.mark.parametrize("model", PRENTIS_MODELS)
def test_prentis_model_cost(model: str):
    from litellm.cost_calculator import cost_per_token

    entry: Final = litellm.model_cost[model]
    prompt_cost, completion_cost = cost_per_token(
        model=model, prompt_tokens=1_000_000, completion_tokens=1_000_000, custom_llm_provider="prentis"
    )

    assert entry["litellm_provider"] == "prentis"
    assert entry["mode"] == "chat"
    assert prompt_cost == pytest.approx(entry["input_cost_per_token"] * 1_000_000)
    assert completion_cost == pytest.approx(entry["output_cost_per_token"] * 1_000_000)
    if "cache_read_input_token_cost" in entry:
        cached_prompt_cost, _ = cost_per_token(
            model=model,
            prompt_tokens=1_000_000,
            completion_tokens=0,
            cache_read_input_tokens=1_000_000,
            custom_llm_provider="prentis",
        )
        assert cached_prompt_cost == pytest.approx(entry["cache_read_input_token_cost"] * 1_000_000)


def test_prentis_backup_cost_map_mirrors_primary():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    prentis_entries: Final = {name: entry for name, entry in cost_map.items() if name.startswith("prentis/")}

    assert prentis_entries
    assert tuple(sorted(prentis_entries)) == PRENTIS_MODELS
    assert prentis_entries == {name: backup[name] for name in prentis_entries}


@pytest.mark.usefixtures("local_model_cost_map")
def test_prentis_wildcard_expands_to_every_prentis_row():
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    assert set(get_known_models_from_wildcard("prentis/*")) == set(PRENTIS_MODELS)


def test_prentis_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    prentis: Final = next(provider for provider in providers if provider["litellm_provider"] == "prentis")

    assert prentis["provider_display_name"] == "Prentis"
    assert prentis["default_model_placeholder"] in PRENTIS_MODELS
    assert {field["key"]: field["required"] for field in prentis["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_prentis_chat_completion_request():
    chat_completion: Final = {
        "id": "chatcmpl_prentis",
        "object": "chat.completion",
        "created": 1_790_000_000,
        "model": "qwen3.8-27b",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Hello from Prentis"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
    }
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=chat_completion)
        response: Final = litellm.completion(
            model=_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="prentis-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer prentis-test-key"
    assert body["model"] == "qwen3.8-27b"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Prentis"
