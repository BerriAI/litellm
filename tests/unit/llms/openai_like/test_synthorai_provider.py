import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_synthorai",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "claude-sonnet-5",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Synthorai"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}


def test_synthorai_is_a_registered_provider():
    assert litellm.LlmProviders.SYNTHORAI.value == "synthorai"
    assert "synthorai" in litellm.provider_list
    assert "synthorai" in litellm.constants.openai_compatible_providers


def test_synthorai_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("SYNTHORAI_API_KEY", "synthorai-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="synthorai/claude-sonnet-5",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "claude-sonnet-5"
    assert provider == "synthorai"
    assert api_key == "synthorai-test-key"
    assert api_base == "https://synthorai.io/v1"


def test_synthorai_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("SYNTHORAI_API_KEY", "synthorai-env-key")
    monkeypatch.setenv("SYNTHORAI_API_BASE", "https://synthorai.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="synthorai/claude-sonnet-5",
        custom_llm_provider=None,
        api_base="https://synthorai.internal.example/v1",
        api_key="synthorai-explicit-key",
    )

    assert provider == "synthorai"
    assert api_key == "synthorai-explicit-key"
    assert api_base == "https://synthorai.internal.example/v1"


def test_synthorai_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("SYNTHORAI_API_KEY", "synthorai-env-key")
    monkeypatch.setenv("SYNTHORAI_API_BASE", "https://synthorai.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="synthorai/claude-sonnet-5",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "synthorai"
    assert api_base == "https://synthorai.env.example/v1"


def test_synthorai_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("SYNTHORAI_API_KEY", "synthorai-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="claude-sonnet-5",
        custom_llm_provider=None,
        api_base="https://synthorai.io/v1",
        api_key=None,
    )

    assert model == "claude-sonnet-5"
    assert provider == "synthorai"
    assert api_key == "synthorai-env-key"
    assert api_base == "https://synthorai.io/v1"


def test_synthorai_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    synthorai = next(provider for provider in providers if provider["litellm_provider"] == "synthorai")

    assert synthorai["provider"] == "SYNTHORAI"
    assert synthorai["provider_display_name"] == "Synthorai"
    assert synthorai["default_model_placeholder"] == "synthorai/claude-sonnet-5"
    assert {field["key"]: field["required"] for field in synthorai["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_synthorai_supported_endpoints():
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

    assert json.loads(backup_path.read_text())["providers"]["synthorai"]["endpoints"] == expected
    assert json.loads(root_path.read_text())["providers"]["synthorai"]["endpoints"] == expected


def test_synthorai_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://synthorai.io/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="synthorai/claude-sonnet-5",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="synthorai-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://synthorai.io/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer synthorai-test-key"
    assert body["model"] == "claude-sonnet-5"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Synthorai"


def test_synthorai_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://synthorai.io/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="synthorai/claude-sonnet-5",
            input="Say hello",
            api_key="synthorai-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer synthorai-test-key"
    assert body["model"] == "claude-sonnet-5"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.output[0].content[0].text == "Hello from Synthorai"


@pytest.mark.asyncio
async def test_synthorai_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://synthorai.io/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model="synthorai/claude-sonnet-5",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="synthorai-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer synthorai-test-key"
    assert body["model"] == "claude-sonnet-5"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert response["content"][0]["text"] == "Hello from Synthorai"
