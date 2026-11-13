import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "https://api.tokenwatt.io/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_UPSTREAM_MODEL: Final = "qwen/qwen3-coder-30b-a3b-instruct"
_MODEL: Final = f"tokenwatt/{_UPSTREAM_MODEL}"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_tokenwatt",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": _UPSTREAM_MODEL,
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from TokenWatt"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}
_TOOL_CALL_COMPLETION: Final = {
    "id": "chatcmpl_tokenwatt_tool",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": _UPSTREAM_MODEL,
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_tokenwatt",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 9, "total_tokens": 29},
}
_WEATHER_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
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
}
TOKENWATT_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("tokenwatt/")))


def test_tokenwatt_is_a_registered_provider():
    assert litellm.LlmProviders.TOKENWATT.value == "tokenwatt"
    assert "tokenwatt" in litellm.provider_list
    assert "tokenwatt" in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints


def test_tokenwatt_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TOKENWATT_API_KEY", "tokenwatt-test-key")
    monkeypatch.delenv("TOKENWATT_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model=_MODEL,
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == _UPSTREAM_MODEL
    assert provider == "tokenwatt"
    assert api_key == "tokenwatt-test-key"
    assert api_base == _DEFAULT_API_BASE


def test_tokenwatt_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TOKENWATT_API_KEY", "tokenwatt-env-key")
    monkeypatch.setenv("TOKENWATT_API_BASE", "https://tokenwatt.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model=_MODEL,
        custom_llm_provider=None,
        api_base="https://tokenwatt.internal.example/v1",
        api_key="tokenwatt-explicit-key",
    )

    assert provider == "tokenwatt"
    assert api_key == "tokenwatt-explicit-key"
    assert api_base == "https://tokenwatt.internal.example/v1"


def test_tokenwatt_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TOKENWATT_API_KEY", "tokenwatt-env-key")
    monkeypatch.setenv("TOKENWATT_API_BASE", "https://tokenwatt.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model=_MODEL,
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "tokenwatt"
    assert api_base == "https://tokenwatt.env.example/v1"


def test_tokenwatt_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TOKENWATT_API_KEY", "tokenwatt-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model=_UPSTREAM_MODEL,
        custom_llm_provider=None,
        api_base=_DEFAULT_API_BASE,
        api_key=None,
    )

    assert model == _UPSTREAM_MODEL
    assert provider == "tokenwatt"
    assert api_key == "tokenwatt-env-key"
    assert api_base == _DEFAULT_API_BASE


@pytest.mark.parametrize("model", TOKENWATT_MODELS)
def test_tokenwatt_model_cost_and_capabilities(model: str):
    from litellm.cost_calculator import cost_per_token
    from litellm.utils import (
        supports_function_calling,
        supports_prompt_caching,
        supports_reasoning,
        supports_response_schema,
        supports_tool_choice,
        supports_vision,
    )

    entry: Final = litellm.model_cost[model]
    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        custom_llm_provider="tokenwatt",
    )
    cached_prompt_cost, _ = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=0,
        cache_read_input_tokens=1_000_000,
        custom_llm_provider="tokenwatt",
    )
    model_info: Final = litellm.get_model_info(model)

    assert prompt_cost == pytest.approx(entry["input_cost_per_token"] * 1_000_000)
    assert completion_cost == pytest.approx(entry["output_cost_per_token"] * 1_000_000)
    assert cached_prompt_cost == pytest.approx(entry["cache_read_input_token_cost"] * 1_000_000)
    assert model_info["max_tokens"] == entry["max_output_tokens"]
    assert model_info["litellm_provider"] == "tokenwatt"
    assert supports_function_calling(model) is entry["supports_function_calling"]
    assert supports_tool_choice(model) is entry["supports_tool_choice"]
    assert supports_reasoning(model) is entry["supports_reasoning"]
    assert supports_prompt_caching(model) is entry["supports_prompt_caching"]
    assert supports_response_schema(model) is entry["supports_response_schema"]
    assert supports_vision(model) is ("image" in entry["supported_modalities"])


@pytest.mark.usefixtures("local_model_cost_map")
def test_tokenwatt_wildcard_expands_to_every_tokenwatt_row():
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    assert set(get_known_models_from_wildcard("tokenwatt/*")) == set(TOKENWATT_MODELS)


def test_tokenwatt_model_missing_from_cost_map_uses_provider_defaults():
    from litellm.utils import supports_function_calling

    assert supports_function_calling("tokenwatt/a-model-not-in-the-cost-map") is True


def test_tokenwatt_backup_registry_mirrors_cost_map():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    tokenwatt_entries: Final = {name: entry for name, entry in cost_map.items() if name.startswith("tokenwatt/")}

    assert tokenwatt_entries
    assert tuple(sorted(tokenwatt_entries)) == TOKENWATT_MODELS
    assert tokenwatt_entries == {name: backup[name] for name in tokenwatt_entries}


def test_tokenwatt_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    tokenwatt: Final = next(provider for provider in providers if provider["litellm_provider"] == "tokenwatt")

    assert tokenwatt["provider"] == "TOKENWATT"
    assert tokenwatt["provider_display_name"] == "TokenWatt"
    assert tokenwatt["default_model_placeholder"] in TOKENWATT_MODELS
    assert {field["key"]: field["required"] for field in tokenwatt["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_tokenwatt_supported_endpoints():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    backup: Final = json.loads(backup_path.read_text())["providers"]["tokenwatt"]["endpoints"]
    root: Final = json.loads(root_path.read_text())["providers"]["tokenwatt"]["endpoints"]

    assert backup == root
    assert {endpoint: root[endpoint] for endpoint in _EXPECTED_ENDPOINTS} == _EXPECTED_ENDPOINTS


def test_tokenwatt_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model=_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=64,
            api_key="tokenwatt-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_COMPLETIONS_URL
    assert request.headers["authorization"] == "Bearer tokenwatt-test-key"
    assert body["model"] == _UPSTREAM_MODEL
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_completion_tokens"] == 64
    assert response.choices[0].message.content == "Hello from TokenWatt"


def test_tokenwatt_tool_call_request_forwards_tools_and_tool_choice():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_TOOL_CALL_COMPLETION)
        response: Final = litellm.completion(
            model=_MODEL,
            messages=[{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
            tools=[_WEATHER_TOOL],
            tool_choice="auto",
            api_key="tokenwatt-test-key",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert body["tools"] == [_WEATHER_TOOL]
    assert body["tool_choice"] == "auto"
    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.tool_calls[0].function.name == "get_weather"
    assert json.loads(response.choices[0].message.tool_calls[0].function.arguments) == {"city": "Paris"}


def test_tokenwatt_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model=_MODEL,
            input="Say hello",
            api_key="tokenwatt-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer tokenwatt-test-key"
    assert body["model"] == _UPSTREAM_MODEL
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    message_items: Final = [item for item in response.output if item.type == "message"]
    assert message_items[0].content[0].text == "Hello from TokenWatt"


@pytest.mark.asyncio
async def test_tokenwatt_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model=_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="tokenwatt-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer tokenwatt-test-key"
    assert body["model"] == _UPSTREAM_MODEL
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert any(block.get("text") == "Hello from TokenWatt" for block in response["content"])


@pytest.fixture
def openai_credentials_but_no_tokenwatt_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("TOKENWATT_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-must-not-leak")
    monkeypatch.setenv("OPENAI_ORGANIZATION", "org-must-not-leak")
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "openai_key", "sk-openai-module-key-must-not-leak")


def test_tokenwatt_missing_key_is_rejected_without_using_the_openai_key(openai_credentials_but_no_tokenwatt_key):
    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        with pytest.raises(litellm.AuthenticationError, match="TOKENWATT_API_KEY"):
            litellm.completion(model=_MODEL, messages=[{"role": "user", "content": "Say hello"}])

    assert route.call_count == 0


def test_tokenwatt_missing_key_is_rejected_on_the_bridged_responses_route(openai_credentials_but_no_tokenwatt_key):
    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        with pytest.raises(litellm.AuthenticationError, match="TOKENWATT_API_KEY"):
            litellm.responses(model=_MODEL, input="Say hello")

    assert route.call_count == 0


def test_tokenwatt_env_key_is_used_and_no_openai_organization_is_sent(
    openai_credentials_but_no_tokenwatt_key, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("TOKENWATT_API_KEY", "tokenwatt-env-key")
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        litellm.completion(model=_MODEL, messages=[{"role": "user", "content": "Say hello"}])

    request: Final = route.calls.last.request
    assert request.headers["authorization"] == "Bearer tokenwatt-env-key"
    assert "openai-organization" not in request.headers
