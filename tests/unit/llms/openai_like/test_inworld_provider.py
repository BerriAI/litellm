import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_inworld",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "inworld/gemma-4-31b-it",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Inworld"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}


def test_inworld_is_a_registered_provider():
    assert litellm.LlmProviders.INWORLD.value == "inworld"
    assert "inworld" in litellm.provider_list
    assert "inworld" in litellm.constants.openai_compatible_providers


def test_inworld_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("INWORLD_API_KEY", "inworld-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="inworld/inworld/models/gemma-4-31b-it",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "inworld/models/gemma-4-31b-it"
    assert provider == "inworld"
    assert api_key == "inworld-test-key"
    assert api_base == "https://api.inworld.ai/v1"


def test_inworld_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("INWORLD_API_KEY", "inworld-env-key")
    monkeypatch.setenv("INWORLD_API_BASE", "https://inworld.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="inworld/inworld/models/gemma-4-31b-it",
        custom_llm_provider=None,
        api_base="https://inworld.internal.example/v1",
        api_key="inworld-explicit-key",
    )

    assert provider == "inworld"
    assert api_key == "inworld-explicit-key"
    assert api_base == "https://inworld.internal.example/v1"


def test_inworld_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("INWORLD_API_KEY", "inworld-env-key")
    monkeypatch.setenv("INWORLD_API_BASE", "https://api.eu.inworld.ai/v1")

    _, provider, _, api_base = get_llm_provider(
        model="inworld/inworld/models/gemma-4-31b-it",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "inworld"
    assert api_base == "https://api.eu.inworld.ai/v1"


def test_inworld_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("INWORLD_API_KEY", "inworld-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="gemma-4-31b-it",
        custom_llm_provider=None,
        api_base="https://api.inworld.ai/v1",
        api_key=None,
    )

    assert model == "gemma-4-31b-it"
    assert provider == "inworld"
    assert api_key == "inworld-env-key"
    assert api_base == "https://api.inworld.ai/v1"


def test_inworld_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    inworld = next(provider for provider in providers if provider["litellm_provider"] == "inworld")

    assert inworld["provider"] == "INWORLD"
    assert inworld["provider_display_name"] == "Inworld"
    assert inworld["default_model_placeholder"] == "inworld/inworld/models/gemma-4-31b-it"
    assert {field["key"]: field["required"] for field in inworld["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_inworld_supported_endpoints():
    expected: Final = {
        "chat_completions": True,
        "messages": True,
        "responses": False,
        "embeddings": False,
        "image_generations": False,
        "audio_transcriptions": True,
        "audio_speech": True,
        "moderations": False,
        "batches": False,
        "rerank": False,
        "a2a": False,
        "interactions": False,
    }
    backup_path = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    root_path = Path(litellm.__file__).parent.parent / "provider_endpoints_support.json"

    assert json.loads(backup_path.read_text())["providers"]["inworld"]["endpoints"] == expected
    assert json.loads(root_path.read_text())["providers"]["inworld"]["endpoints"] == expected


def test_inworld_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.inworld.ai/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="inworld/inworld/models/gemma-4-31b-it",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="inworld-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.inworld.ai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer inworld-test-key"
    # Inworld-hosted model IDs live under the "inworld/" namespace upstream; only LiteLLM's routing prefix is removed.
    assert body["model"] == "inworld/models/gemma-4-31b-it"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Inworld"


def test_inworld_routes_third_party_model_ids_unchanged():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.inworld.ai/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        litellm.completion(
            model="inworld/openai/gpt-5-mini",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="inworld-test-key",
        )

    assert json.loads(route.calls.last.request.content)["model"] == "openai/gpt-5-mini"


@pytest.mark.asyncio
async def test_inworld_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.inworld.ai/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model="inworld/inworld/models/gemma-4-31b-it",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="inworld-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer inworld-test-key"
    assert body["model"] == "inworld/models/gemma-4-31b-it"
    assert body["max_tokens"] == 32
    assert response["content"][0]["text"] == "Hello from Inworld"


def test_inworld_speech_uses_provider_api_key_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INWORLD_API_KEY", "inworld-env-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.inworld.ai/v1/audio/speech").respond(
            200, content=b"ID3-audio", headers={"Content-Type": "audio/mpeg"}
        )
        response: Final = litellm.speech(
            model="inworld/inworld-tts-2",
            voice="Dennis",
            input="Hello from Inworld",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer inworld-env-key"
    assert body["model"] == "inworld-tts-2"
    assert body["voice"] == "Dennis"
    assert body["input"] == "Hello from Inworld"
    assert response.content == b"ID3-audio"


def test_inworld_transcription_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("INWORLD_API_KEY", "inworld-env-key")
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.inworld.ai/v1/audio/transcriptions").respond(
            200, json={"text": "Hello from Inworld"}
        )
        response: Final = litellm.transcription(
            model="inworld/inworld/inworld-stt-1",
            file=("hello.mp3", b"ID3-audio", "audio/mpeg"),
        )

    request: Final = route.calls.last.request
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer inworld-env-key"
    assert b'name="model"\r\n\r\ninworld/inworld-stt-1' in request.content
    assert response.text == "Hello from Inworld"
