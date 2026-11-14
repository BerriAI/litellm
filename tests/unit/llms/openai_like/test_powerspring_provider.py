import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "https://api.powerspring.eu/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_powerspring",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "gpt-oss-120b",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Powerspring"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}
_GET_WEATHER_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
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
POWERSPRING_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("powerspring/")))


def test_powerspring_is_a_registered_provider():
    assert litellm.LlmProviders.POWERSPRING.value == "powerspring"
    assert "powerspring" in litellm.provider_list
    assert "powerspring" in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints


def test_powerspring_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("POWERSPRING_API_KEY", "powerspring-test-key")
    monkeypatch.delenv("POWERSPRING_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="powerspring/gpt-oss-120b",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "gpt-oss-120b"
    assert provider == "powerspring"
    assert api_key == "powerspring-test-key"
    assert api_base == _DEFAULT_API_BASE


def test_powerspring_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("POWERSPRING_API_KEY", "powerspring-env-key")
    monkeypatch.setenv("POWERSPRING_API_BASE", "https://powerspring.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="powerspring/gpt-oss-120b",
        custom_llm_provider=None,
        api_base="https://powerspring.internal.example/v1",
        api_key="powerspring-explicit-key",
    )

    assert provider == "powerspring"
    assert api_key == "powerspring-explicit-key"
    assert api_base == "https://powerspring.internal.example/v1"


def test_powerspring_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("POWERSPRING_API_KEY", "powerspring-env-key")
    monkeypatch.setenv("POWERSPRING_API_BASE", "https://powerspring.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="powerspring/gpt-oss-120b",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "powerspring"
    assert api_base == "https://powerspring.env.example/v1"


@pytest.mark.parametrize("model", POWERSPRING_MODELS)
def test_powerspring_model_capabilities(model: str):
    from litellm.utils import (
        supports_function_calling,
        supports_reasoning,
        supports_response_schema,
        supports_tool_choice,
        supports_vision,
    )

    entry: Final = litellm.model_cost[model]
    model_info: Final = litellm.get_model_info(model)

    assert model_info["litellm_provider"] == "powerspring"
    assert model_info["max_input_tokens"] == entry["max_input_tokens"]
    assert model_info["max_output_tokens"] == entry["max_output_tokens"]
    assert supports_function_calling(model) is entry["supports_function_calling"]
    assert supports_tool_choice(model) is entry["supports_tool_choice"]
    assert supports_reasoning(model) is entry["supports_reasoning"]
    assert supports_response_schema(model) is entry["supports_response_schema"]
    assert supports_vision(model) is ("image" in entry["supported_modalities"])


@pytest.mark.usefixtures("local_model_cost_map")
def test_powerspring_wildcard_expands_to_every_powerspring_row():
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    assert set(get_known_models_from_wildcard("powerspring/*")) == set(POWERSPRING_MODELS)


def test_powerspring_model_missing_from_cost_map_uses_provider_defaults():
    from litellm.utils import supports_function_calling, supports_reasoning

    assert supports_function_calling("powerspring/a-model-not-in-the-cost-map") is True
    assert supports_reasoning("powerspring/a-model-not-in-the-cost-map") is True


def test_powerspring_backup_registry_mirrors_cost_map():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    powerspring_entries: Final = {name: entry for name, entry in cost_map.items() if name.startswith("powerspring/")}

    assert powerspring_entries
    assert tuple(sorted(powerspring_entries)) == POWERSPRING_MODELS
    assert powerspring_entries == {name: backup[name] for name in powerspring_entries}


def test_powerspring_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    powerspring: Final = next(provider for provider in providers if provider["litellm_provider"] == "powerspring")

    assert powerspring["provider"] == "POWERSPRING"
    assert powerspring["provider_display_name"] == "Powerspring"
    assert powerspring["default_model_placeholder"] in POWERSPRING_MODELS
    assert {field["key"]: field["required"] for field in powerspring["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_powerspring_supported_endpoints():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    backup: Final = json.loads(backup_path.read_text())["providers"]["powerspring"]["endpoints"]
    root: Final = json.loads(root_path.read_text())["providers"]["powerspring"]["endpoints"]

    assert backup == root
    assert {endpoint: root[endpoint] for endpoint in _EXPECTED_ENDPOINTS} == _EXPECTED_ENDPOINTS


@pytest.mark.parametrize("model", ["powerspring/gpt-oss-120b", "powerspring/a-model-not-in-the-cost-map"])
def test_powerspring_chat_completion_request_forwards_tools(model: str):
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Weather in Barcelona?"}],
            tools=[_GET_WEATHER_TOOL],
            tool_choice="auto",
            max_completion_tokens=64,
            reasoning_effort="none",
            api_key="powerspring-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_COMPLETIONS_URL
    assert request.headers["authorization"] == "Bearer powerspring-test-key"
    assert body["model"] == model.removeprefix("powerspring/")
    assert body["tools"] == [_GET_WEATHER_TOOL]
    assert body["tool_choice"] == "auto"
    assert body["max_completion_tokens"] == 64
    assert body["reasoning_effort"] == "none"
    assert response.choices[0].message.content == "Hello from Powerspring"


def test_powerspring_streaming_chat_completion_request():
    chunks: Final = (
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello "}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "from Powerspring"}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}},
    )
    envelope: Final = {
        "id": "chatcmpl_powerspring",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-oss-120b",
    }
    events: Final = tuple(f"data: {json.dumps(envelope | chunk)}\n\n" for chunk in chunks)
    stream_body: Final = "".join(events) + "data: [DONE]\n\n"
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(
            200, text=stream_body, headers={"content-type": "text/event-stream"}
        )
        response: Final = litellm.completion(
            model="powerspring/gpt-oss-120b",
            messages=[{"role": "user", "content": "Say hello"}],
            stream=True,
            stream_options={"include_usage": True},
            api_key="powerspring-test-key",
        )
        received: Final = tuple(response)

    body: Final = json.loads(route.calls.last.request.content)
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert "".join(chunk.choices[0].delta.content or "" for chunk in received if chunk.choices) == (
        "Hello from Powerspring"
    )
    assert received[-1].usage.total_tokens == 7


def test_powerspring_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_DEFAULT_API_BASE}/responses").respond(
            200,
            json={
                "id": "resp_powerspring",
                "object": "response",
                "created_at": 1_790_000_000,
                "model": "gpt-oss-120b",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_powerspring",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from Powerspring", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="powerspring/gpt-oss-120b",
            input="Say hello",
            api_key="powerspring-test-key",
            store=True,
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == f"{_DEFAULT_API_BASE}/responses"
    assert request.headers["authorization"] == "Bearer powerspring-test-key"
    assert body["model"] == "gpt-oss-120b"
    assert body["input"] == "Say hello"
    assert body["store"] is False
    assert response.output[0].content[0].text == "Hello from Powerspring"


@pytest.mark.asyncio
async def test_powerspring_anthropic_messages_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_DEFAULT_API_BASE}/messages").respond(
            200,
            json={
                "id": "msg_powerspring",
                "type": "message",
                "role": "assistant",
                "model": "gpt-oss-120b",
                "content": [{"type": "text", "text": "Hello from Powerspring"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="powerspring/gpt-oss-120b",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="powerspring-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == f"{_DEFAULT_API_BASE}/messages"
    assert request.headers["authorization"] == "Bearer powerspring-test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "gpt-oss-120b"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from Powerspring"


def test_missing_powerspring_key_does_not_send_openai_credentials_on_chat(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("POWERSPRING_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-key")
    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        with pytest.raises(litellm.AuthenticationError, match="POWERSPRING_API_KEY"):
            litellm.completion(model="powerspring/gpt-oss-120b", messages=[{"role": "user", "content": "Say hello"}])

    assert route.call_count == 0


def test_chat_does_not_send_the_openai_organization_to_powerspring(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_ORGANIZATION", "org-openai-only")
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        litellm.completion(
            model="powerspring/gpt-oss-120b",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="powerspring-test-key",
        )

    assert "openai-organization" not in route.calls.last.request.headers


def test_missing_powerspring_key_does_not_send_openai_credentials_on_responses(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("POWERSPRING_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-key")
    monkeypatch.setenv("OPENAI_ORGANIZATION", "org-openai-only")
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_DEFAULT_API_BASE}/responses").respond(
            401, json={"error": {"message": "no key"}}
        )
        with pytest.raises(litellm.AuthenticationError):
            litellm.responses(model="powerspring/gpt-oss-120b", input="Say hello")

    headers: Final = route.calls.last.request.headers
    assert "unrelated-openai-key" not in headers.get("authorization", "")
    assert "openai-organization" not in headers


@pytest.mark.asyncio
async def test_missing_powerspring_key_does_not_send_openai_credentials_on_messages(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("POWERSPRING_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unrelated-anthropic-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_DEFAULT_API_BASE}/messages").respond(
            401, json={"error": {"message": "no key"}}
        )
        with pytest.raises(litellm.AuthenticationError):
            await litellm.anthropic.messages.acreate(
                model="powerspring/gpt-oss-120b",
                messages=[{"role": "user", "content": "Say hello"}],
                max_tokens=32,
            )

    headers: Final = route.calls.last.request.headers
    assert "unrelated-openai-key" not in headers.get("authorization", "")
    assert "unrelated-anthropic-key" not in headers.get("x-api-key", "")
    assert "unrelated-anthropic-key" not in headers.get("authorization", "")
