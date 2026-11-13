import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "https://api.bineric.com/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_bineric",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "claude-sonnet-5.5",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Bineric"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}
_EXPECTED_ENDPOINTS: Final = {
    "chat_completions": True,
    "messages": False,
    "responses": False,
    "embeddings": False,
    "image_generations": False,
    "audio_transcriptions": False,
    "audio_speech": False,
    "moderations": False,
    "batches": False,
    "rerank": False,
    "systemone": True,
}
BINERIC_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("bineric/")))
BINERIC_CHAT_MODELS: Final = tuple(name for name in BINERIC_MODELS if litellm.model_cost[name]["mode"] == "chat")
BINERIC_DECISION_MODELS: Final = tuple(
    name for name in BINERIC_MODELS if litellm.model_cost[name]["mode"] == "evaluation"
)


def test_bineric_is_a_registered_provider():
    assert litellm.LlmProviders.BINERIC.value == "bineric"
    assert "bineric" in litellm.provider_list
    assert "bineric" in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints


def test_bineric_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BINERIC_API_KEY", "bineric-test-key")
    monkeypatch.delenv("BINERIC_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="bineric/claude-sonnet-5.5",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "claude-sonnet-5.5"
    assert provider == "bineric"
    assert api_key == "bineric-test-key"
    assert api_base == _DEFAULT_API_BASE


def test_bineric_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BINERIC_API_KEY", "bineric-env-key")
    monkeypatch.setenv("BINERIC_API_BASE", "https://bineric.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="bineric/claude-sonnet-5.5",
        custom_llm_provider=None,
        api_base="https://bineric.internal.example/v1",
        api_key="bineric-explicit-key",
    )

    assert provider == "bineric"
    assert api_key == "bineric-explicit-key"
    assert api_base == "https://bineric.internal.example/v1"


def test_bineric_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BINERIC_API_KEY", "bineric-env-key")
    monkeypatch.setenv("BINERIC_API_BASE", "https://bineric.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="bineric/claude-sonnet-5.5",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "bineric"
    assert api_base == "https://bineric.env.example/v1"


def test_bineric_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BINERIC_API_KEY", "bineric-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="claude-sonnet-5.5",
        custom_llm_provider=None,
        api_base=_DEFAULT_API_BASE,
        api_key=None,
    )

    assert model == "claude-sonnet-5.5"
    assert provider == "bineric"
    assert api_key == "bineric-env-key"
    assert api_base == _DEFAULT_API_BASE


@pytest.mark.parametrize("model", BINERIC_CHAT_MODELS)
def test_bineric_model_cost_and_capabilities(model: str):
    from litellm.cost_calculator import cost_per_token
    from litellm.utils import supports_function_calling, supports_vision

    entry: Final = litellm.model_cost[model]
    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        custom_llm_provider="bineric",
    )
    model_info: Final = litellm.get_model_info(model)

    assert entry["input_cost_per_token"] > 0
    assert prompt_cost == pytest.approx(entry["input_cost_per_token"] * 1_000_000)
    assert completion_cost == pytest.approx(entry["output_cost_per_token"] * 1_000_000)
    assert model_info["max_tokens"] == entry.get("max_output_tokens")
    assert model_info["max_input_tokens"] == entry.get("max_input_tokens")
    assert model_info["litellm_provider"] == "bineric"
    if entry.get("supports_function_calling"):
        assert supports_function_calling(model) is True
    if entry.get("supports_vision"):
        assert supports_vision(model) is True


@pytest.mark.parametrize("model", BINERIC_DECISION_MODELS)
def test_bineric_zuse_rows_are_priced_per_input_token_on_systemone(model: str):
    entry: Final = litellm.model_cost[model]

    assert model.startswith("bineric/zuse-")
    assert entry["supported_endpoints"] == ["/v1/systemone"]
    assert entry["input_cost_per_token"] > 0
    assert entry["output_cost_per_token"] == 0.0


def test_bineric_ships_chat_and_decision_models():
    assert BINERIC_CHAT_MODELS
    assert BINERIC_DECISION_MODELS
    assert set(BINERIC_CHAT_MODELS) | set(BINERIC_DECISION_MODELS) == set(BINERIC_MODELS)


@pytest.mark.usefixtures("local_model_cost_map")
def test_bineric_wildcard_expands_to_every_bineric_row():
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    assert set(get_known_models_from_wildcard("bineric/*")) == set(BINERIC_MODELS)


def test_bineric_backup_registry_mirrors_cost_map():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    bineric_entries: Final = {name: entry for name, entry in cost_map.items() if name.startswith("bineric/")}

    assert bineric_entries
    assert tuple(sorted(bineric_entries)) == BINERIC_MODELS
    assert bineric_entries == {name: backup[name] for name in bineric_entries}


def test_bineric_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    bineric: Final = next(provider for provider in providers if provider["litellm_provider"] == "bineric")

    assert bineric["provider"] == "BINERIC"
    assert bineric["provider_display_name"] == "Bineric"
    assert bineric["default_model_placeholder"] in BINERIC_MODELS
    assert {field["key"]: field["required"] for field in bineric["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_bineric_supported_endpoints():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    backup: Final = json.loads(backup_path.read_text())["providers"]["bineric"]["endpoints"]
    root: Final = json.loads(root_path.read_text())["providers"]["bineric"]["endpoints"]

    assert backup == root
    assert {endpoint: root[endpoint] for endpoint in _EXPECTED_ENDPOINTS} == _EXPECTED_ENDPOINTS


def test_bineric_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="bineric/claude-sonnet-5.5",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=64,
            api_key="bineric-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_COMPLETIONS_URL
    assert request.headers["authorization"] == "Bearer bineric-test-key"
    assert body["model"] == "claude-sonnet-5.5"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_completion_tokens"] == 64
    assert response.choices[0].message.content == "Hello from Bineric"


@pytest.mark.parametrize(
    ("usage_extra", "expected_cost"),
    (({"cost": 0.18}, 0.18), ({}, None)),
    ids=("reported_cost_is_billed", "no_reported_cost"),
)
def test_bineric_per_request_model_without_a_cost_row_still_routes(
    usage_extra: dict[str, float],
    expected_cost: float | None,
):
    assert "bineric/speech-to-text" not in litellm.model_cost
    completion: Final = {
        **_CHAT_COMPLETION,
        "model": "speech-to-text",
        "usage": {**_CHAT_COMPLETION["usage"], **usage_extra},
    }
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=completion)
        response: Final = litellm.completion(
            model="bineric/speech-to-text",
            messages=[{"role": "user", "content": "Transcribe this"}],
            api_key="bineric-test-key",
        )

    assert route.call_count == 1
    assert json.loads(route.calls.last.request.content)["model"] == "speech-to-text"
    if expected_cost is None:
        assert response._hidden_params["response_cost"] is None
    else:
        assert response._hidden_params["response_cost"] == pytest.approx(expected_cost)
