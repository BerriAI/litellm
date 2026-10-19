"""
Tests for the AI Stupid Level provider (JSON-configured, OpenAI-compatible).

AI Stupid Level is a router: the model id is a routing mode such as ``auto-coding``,
or a real model id to pin one model.
"""

import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_BASE: Final = "https://aistupidlevel.info/v1"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_asl",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "claude-opus-5-5",
    "system_fingerprint": "aism_v1_anthropic",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from the router"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8},
}


def test_aistupidlevel_is_a_registered_provider():
    assert litellm.LlmProviders.AISTUPIDLEVEL.value == "aistupidlevel"
    assert "aistupidlevel" in litellm.provider_list
    assert "aistupidlevel" in litellm.constants.openai_compatible_providers
    assert _BASE in litellm.constants.openai_compatible_endpoints


def test_aistupidlevel_json_config():
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    config = JSONProviderRegistry.get("aistupidlevel")
    assert config is not None
    assert config.base_url == _BASE
    assert config.api_key_env == "AISTUPIDLEVEL_API_KEY"
    assert config.api_base_env == "AISTUPIDLEVEL_API_BASE"
    # The router accepts max_completion_tokens as sent, so nothing is remapped.
    assert config.param_mappings == {}


def test_aistupidlevel_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AISTUPIDLEVEL_API_KEY", "aism-env-key")
    monkeypatch.delenv("AISTUPIDLEVEL_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(model="aistupidlevel/auto-coding")

    assert model == "auto-coding"
    assert provider == "aistupidlevel"
    assert api_key == "aism-env-key"
    assert api_base == _BASE


def test_aistupidlevel_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AISTUPIDLEVEL_API_KEY", "aism-env-key")
    monkeypatch.setenv("AISTUPIDLEVEL_API_BASE", "https://router.example/v1")

    _, provider, _, api_base = get_llm_provider(model="aistupidlevel/auto-reasoning")

    assert provider == "aistupidlevel"
    assert api_base == "https://router.example/v1"


def test_aistupidlevel_api_base_autodetects_provider():
    _, provider, _, api_base = get_llm_provider(model="auto-coding", api_base=_BASE)

    assert provider == "aistupidlevel"
    assert api_base == _BASE


def test_aistupidlevel_routed_models_support_tools():
    """Every routed or pinned model can call tools, so tool params must not be dropped."""
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    assert litellm.supports_function_calling(model="aistupidlevel/auto-tooling") is True

    config = create_config_class(JSONProviderRegistry.get("aistupidlevel"))()
    supported: Final = config.get_supported_openai_params("auto-tooling")
    assert "tools" in supported
    assert "tool_choice" in supported


def test_aistupidlevel_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    entry = next(p for p in providers if p["litellm_provider"] == "aistupidlevel")

    assert entry["provider_display_name"] == "AI Stupid Level"
    assert entry["default_model_placeholder"] == "aistupidlevel/auto-coding"
    assert {field["key"]: field["required"] for field in entry["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_aistupidlevel_supported_endpoints():
    expected: Final = {
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
    backup_path = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    assert json.loads(backup_path.read_text())["providers"]["aistupidlevel"]["endpoints"] == expected
    assert json.loads(root_path.read_text())["providers"]["aistupidlevel"]["endpoints"] == expected


def test_aistupidlevel_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_BASE}/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="aistupidlevel/auto-coding",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=64,
            api_key="aism-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == f"{_BASE}/chat/completions"
    assert request.headers["authorization"] == "Bearer aism-test-key"
    assert body["model"] == "auto-coding"
    assert body["max_completion_tokens"] == 64
    assert response.choices[0].message.content == "Hello from the router"


def test_aistupidlevel_tools_are_forwarded():
    tools: Final = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_BASE}/chat/completions").respond(200, json=_CHAT_COMPLETION)
        litellm.completion(
            model="aistupidlevel/auto-tooling",
            messages=[{"role": "user", "content": "Weather in Oslo?"}],
            tools=tools,
            tool_choice="required",
            api_key="aism-test-key",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert body["tools"] == tools
    assert body["tool_choice"] == "required"


def test_aistupidlevel_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_BASE}/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="aistupidlevel/auto-best",
            input="Say hello",
            api_key="aism-test-key",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert route.call_count == 1
    assert body["model"] == "auto-best"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.output[0].content[0].text == "Hello from the router"


@pytest.mark.asyncio
async def test_aistupidlevel_anthropic_messages_request_is_bridged_to_chat_completions(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_BASE}/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model="aistupidlevel/auto-coding",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="aism-test-key",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert route.call_count == 1
    assert route.calls.last.request.headers["authorization"] == "Bearer aism-test-key"
    assert body["model"] == "auto-coding"
    assert body["max_tokens"] == 32
    assert response["content"][0]["text"] == "Hello from the router"
