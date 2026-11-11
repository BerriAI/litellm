import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "https://api.openany.si/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_openany",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "MiniMax-M2.5",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from OpenAny"},
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
}


def test_openany_is_a_registered_provider():
    assert litellm.LlmProviders.OPENANY.value == "openany"
    assert "openany" in litellm.provider_list
    assert "openany" in litellm.constants.openai_compatible_providers
    assert _DEFAULT_API_BASE in litellm.constants.openai_compatible_endpoints


def test_openany_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENANY_API_KEY", "openany-test-key")
    monkeypatch.delenv("OPENANY_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="openany/MiniMax-M2.5",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "MiniMax-M2.5"
    assert provider == "openany"
    assert api_key == "openany-test-key"
    assert api_base == _DEFAULT_API_BASE


def test_openany_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENANY_API_KEY", "openany-env-key")
    monkeypatch.setenv("OPENANY_API_BASE", "https://openany.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="openany/MiniMax-M2.5",
        custom_llm_provider=None,
        api_base="https://openany.internal.example/v1",
        api_key="openany-explicit-key",
    )

    assert provider == "openany"
    assert api_key == "openany-explicit-key"
    assert api_base == "https://openany.internal.example/v1"


def test_openany_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENANY_API_KEY", "openany-env-key")
    monkeypatch.setenv("OPENANY_API_BASE", "https://openany.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="openany/MiniMax-M2.5",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "openany"
    assert api_base == "https://openany.env.example/v1"


def test_openany_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENANY_API_KEY", "openany-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="MiniMax-M2.5",
        custom_llm_provider=None,
        api_base=_DEFAULT_API_BASE,
        api_key=None,
    )

    assert model == "MiniMax-M2.5"
    assert provider == "openany"
    assert api_key == "openany-env-key"
    assert api_base == _DEFAULT_API_BASE


def test_openany_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    openany: Final = next(provider for provider in providers if provider["litellm_provider"] == "openany")

    assert openany["provider"] == "OPENANY"
    assert openany["provider_display_name"] == "OpenAny"
    assert openany["default_model_placeholder"] == "openany/MiniMax-M2.5"
    assert {field["key"]: field["required"] for field in openany["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_openany_supported_endpoints():
    backup_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path: Final = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    backup: Final = json.loads(backup_path.read_text())["providers"]["openany"]["endpoints"]
    root: Final = json.loads(root_path.read_text())["providers"]["openany"]["endpoints"]

    assert backup == root
    assert {endpoint: root[endpoint] for endpoint in _EXPECTED_ENDPOINTS} == _EXPECTED_ENDPOINTS


def test_openany_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="openany/MiniMax-M2.5",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=64,
            api_key="openany-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_COMPLETIONS_URL
    assert request.headers["authorization"] == "Bearer openany-test-key"
    assert body["model"] == "MiniMax-M2.5"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 64
    assert response.choices[0].message.content == "Hello from OpenAny"


def test_openany_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="openany/MiniMax-M2.5",
            input="Say hello",
            api_key="openany-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer openany-test-key"
    assert body["model"] == "MiniMax-M2.5"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    message_items: Final = [item for item in response.output if item.type == "message"]
    assert message_items[0].content[0].text == "Hello from OpenAny"


@pytest.mark.asyncio
async def test_openany_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model="openany/MiniMax-M2.5",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="openany-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _CHAT_COMPLETIONS_URL
    assert request.headers["authorization"] == "Bearer openany-test-key"
    assert body["model"] == "MiniMax-M2.5"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert any(block.get("text") == "Hello from OpenAny" for block in response["content"])
