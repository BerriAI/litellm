import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "https://router.inference.ai.exchange/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_architect",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "deepseek-v4.1-flash",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Architect"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}
_EXPECTED_ENDPOINTS: Final = {
    "chat_completions": True,
    "messages": True,
    "responses": True,
    "text_completion": False,
    "embeddings": False,
    "image_generations": False,
    "audio_transcriptions": False,
    "audio_speech": False,
    "moderations": False,
    "batches": False,
    "rerank": False,
}


def test_architect_is_a_registered_provider():
    assert litellm.LlmProviders.ARCHITECT.value == "architect"
    assert "architect" in litellm.provider_list
    assert "architect" in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints


def test_architect_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ARCHITECT_API_KEY", "architect-test-key")
    monkeypatch.delenv("ARCHITECT_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="architect/deepseek-v4.1-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "deepseek-v4.1-flash"
    assert provider == "architect"
    assert api_key == "architect-test-key"
    assert api_base == _DEFAULT_API_BASE


def test_architect_keeps_a_seller_qualified_model_name(monkeypatch: pytest.MonkeyPatch):
    """The exchange names one seller's offer as `<seller>/<model>`, so only the first prefix is removed."""
    monkeypatch.setenv("ARCHITECT_API_KEY", "architect-test-key")

    model, provider, _, _ = get_llm_provider(
        model="architect/sokkan/inkling-2-mini",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "sokkan/inkling-2-mini"
    assert provider == "architect"


def test_architect_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ARCHITECT_API_KEY", "architect-env-key")
    monkeypatch.setenv("ARCHITECT_API_BASE", "https://architect.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="architect/deepseek-v4.1-flash",
        custom_llm_provider=None,
        api_base="https://architect.internal.example/v1",
        api_key="architect-explicit-key",
    )

    assert provider == "architect"
    assert api_key == "architect-explicit-key"
    assert api_base == "https://architect.internal.example/v1"


def test_architect_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ARCHITECT_API_KEY", "architect-env-key")
    monkeypatch.setenv("ARCHITECT_API_BASE", "https://architect.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="architect/deepseek-v4.1-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "architect"
    assert api_base == "https://architect.env.example/v1"


def test_architect_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ARCHITECT_API_KEY", "architect-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="deepseek-v4.1-flash",
        custom_llm_provider=None,
        api_base=_DEFAULT_API_BASE,
        api_key=None,
    )

    assert model == "deepseek-v4.1-flash"
    assert provider == "architect"
    assert api_key == "architect-env-key"
    assert api_base == _DEFAULT_API_BASE


def test_architect_model_uses_provider_defaults():
    from litellm.utils import supports_function_calling

    assert supports_function_calling("architect/deepseek-v4.1-flash") is True


def test_architect_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    architect: Final = next(provider for provider in providers if provider["litellm_provider"] == "architect")

    assert architect["provider"] == "ARCHITECT"
    assert architect["provider_display_name"] == "Architect"
    assert architect["default_model_placeholder"] == "architect/deepseek-v4.1-flash"
    assert {field["key"]: field["required"] for field in architect["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_architect_supported_endpoints():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    backup: Final = json.loads(backup_path.read_text())["providers"]["architect"]["endpoints"]
    root: Final = json.loads(root_path.read_text())["providers"]["architect"]["endpoints"]

    assert backup == root
    assert {endpoint: root[endpoint] for endpoint in _EXPECTED_ENDPOINTS} == _EXPECTED_ENDPOINTS


def test_architect_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="architect/deepseek-v4.1-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=64,
            api_key="architect-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_COMPLETIONS_URL
    assert request.headers["authorization"] == "Bearer architect-test-key"
    assert body["model"] == "deepseek-v4.1-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_completion_tokens"] == 64
    assert response.choices[0].message.content == "Hello from Architect"


def test_architect_chat_completion_sends_tools():
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
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        litellm.completion(
            model="architect/deepseek-v4.1-flash",
            messages=[{"role": "user", "content": "Weather in Paris?"}],
            tools=tools,
            tool_choice="auto",
            api_key="architect-test-key",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert body["tools"] == tools
    assert body["tool_choice"] == "auto"


def test_architect_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="architect/deepseek-v4.1-flash",
            input="Say hello",
            api_key="architect-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer architect-test-key"
    assert body["model"] == "deepseek-v4.1-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    message_items: Final = [item for item in response.output if item.type == "message"]
    assert message_items[0].content[0].text == "Hello from Architect"


@pytest.mark.asyncio
async def test_architect_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model="architect/deepseek-v4.1-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="architect-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer architect-test-key"
    assert body["model"] == "deepseek-v4.1-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert any(block.get("text") == "Hello from Architect" for block in response["content"])
