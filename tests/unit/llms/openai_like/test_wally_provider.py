import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "https://inference.runanywhere.ai/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_wally",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "glm-5.3-flash",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Wally", "reasoning_content": "Say hi"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}
_EXPECTED_ENDPOINTS: Final = {
    "chat_completions": True,
    "messages": True,
    "responses": True,
    "embeddings": False,
    "image_generations": False,
    "audio_transcriptions": False,
    "audio_speech": False,
    "moderations": False,
    "batches": False,
    "rerank": False,
    "a2a": False,
    "interactions": False,
}
WALLY_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("wally/")))


def test_wally_is_a_registered_provider():
    assert litellm.LlmProviders.WALLY.value == "wally"
    assert "wally" in litellm.provider_list
    assert "wally" in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints


def test_wally_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WALLY_API_KEY", "wally-test-key")
    monkeypatch.delenv("WALLY_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="wally/glm-5.3-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "glm-5.3-flash"
    assert provider == "wally"
    assert api_key == "wally-test-key"
    assert api_base == _DEFAULT_API_BASE


def test_wally_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WALLY_API_KEY", "wally-env-key")
    monkeypatch.setenv("WALLY_API_BASE", "https://wally.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="wally/glm-5.3-flash",
        custom_llm_provider=None,
        api_base="https://wally.internal.example/v1",
        api_key="wally-explicit-key",
    )

    assert provider == "wally"
    assert api_key == "wally-explicit-key"
    assert api_base == "https://wally.internal.example/v1"


def test_wally_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WALLY_API_KEY", "wally-env-key")
    monkeypatch.setenv("WALLY_API_BASE", "https://wally.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="wally/glm-5.3-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "wally"
    assert api_base == "https://wally.env.example/v1"


def test_wally_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("WALLY_API_KEY", "wally-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="glm-5.3-flash",
        custom_llm_provider=None,
        api_base=_DEFAULT_API_BASE,
        api_key=None,
    )

    assert model == "glm-5.3-flash"
    assert provider == "wally"
    assert api_key == "wally-env-key"
    assert api_base == _DEFAULT_API_BASE


@pytest.mark.parametrize("model", WALLY_MODELS)
def test_wally_model_cost_and_capabilities(model: str):
    from litellm.cost_calculator import cost_per_token

    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        custom_llm_provider="wally",
    )
    model_info: Final = litellm.get_model_info(model)

    assert prompt_cost == pytest.approx(model_info["input_cost_per_token"] * 1_000_000)
    assert completion_cost == pytest.approx(model_info["output_cost_per_token"] * 1_000_000)
    assert 0 < model_info["cache_read_input_token_cost"] < model_info["input_cost_per_token"]
    assert model_info["input_cost_per_token"] < model_info["output_cost_per_token"]
    assert model_info["max_tokens"] == model_info["max_output_tokens"] <= model_info["max_input_tokens"]
    assert model_info["litellm_provider"] == "wally"
    assert model_info["mode"] == "chat"
    assert model_info["supports_function_calling"] is True
    assert model_info["supports_tool_choice"] is True
    assert model_info["supports_reasoning"] is True
    assert model_info["supports_response_schema"] is True
    assert litellm.supports_vision(model) is ("image" in litellm.model_cost[model]["supported_modalities"])


def test_wally_backup_registry_mirrors_cost_map():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    wally_entries: Final = {name: entry for name, entry in cost_map.items() if name.startswith("wally/")}

    assert wally_entries
    assert tuple(sorted(wally_entries)) == WALLY_MODELS
    assert wally_entries == {name: backup[name] for name in wally_entries}


def test_wally_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    wally: Final = next(provider for provider in providers if provider["litellm_provider"] == "wally")

    assert wally["provider"] == "WALLY"
    assert wally["provider_display_name"] == "Wally"
    assert wally["default_model_placeholder"] in WALLY_MODELS
    assert {field["key"]: field["required"] for field in wally["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_wally_supported_endpoints():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    assert json.loads(backup_path.read_text())["providers"]["wally"]["endpoints"] == _EXPECTED_ENDPOINTS
    assert json.loads(root_path.read_text())["providers"]["wally"]["endpoints"] == _EXPECTED_ENDPOINTS


@pytest.mark.parametrize("reasoning_effort", ["none", "max"])
def test_wally_chat_completion_request(reasoning_effort: str):
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="wally/glm-5.3-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=64,
            reasoning_effort=reasoning_effort,
            api_key="wally-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_COMPLETIONS_URL
    assert request.headers["authorization"] == "Bearer wally-test-key"
    assert body["model"] == "glm-5.3-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_completion_tokens"] == 64
    assert body["reasoning_effort"] == reasoning_effort
    assert response.choices[0].message.content == "Hello from Wally"
    assert response.choices[0].message.reasoning_content == "Say hi"


def test_wally_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="wally/glm-5.3-flash",
            input="Say hello",
            api_key="wally-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer wally-test-key"
    assert body["model"] == "glm-5.3-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    message_items: Final = [item for item in response.output if item.type == "message"]
    assert message_items[0].content[0].text == "Hello from Wally"


@pytest.mark.asyncio
async def test_wally_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model="wally/glm-5.3-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="wally-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer wally-test-key"
    assert body["model"] == "glm-5.3-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert any(block.get("text") == "Hello from Wally" for block in response["content"])
