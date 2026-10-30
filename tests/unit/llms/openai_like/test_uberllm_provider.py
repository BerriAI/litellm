import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache


def test_uberllm_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("UBERLLM_API_KEY", "uberllm-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="uberllm/deepseek/deepseek-v4.1-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "deepseek/deepseek-v4.1-flash"
    assert provider == "uberllm"
    assert api_key == "uberllm-test-key"
    assert api_base == "https://api.uberllm.dev/v1"


def test_uberllm_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("UBERLLM_API_KEY", "uberllm-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="uberllm/deepseek/deepseek-v4.1-flash",
        custom_llm_provider=None,
        api_base="https://uberllm.internal.example/v1",
        api_key="uberllm-explicit-key",
    )

    assert provider == "uberllm"
    assert api_key == "uberllm-explicit-key"
    assert api_base == "https://uberllm.internal.example/v1"


def test_uberllm_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    uberllm = next(provider for provider in providers if provider["litellm_provider"] == "uberllm")

    assert uberllm["provider"] == "UBERLLM"
    assert uberllm["provider_display_name"] == "UberLLM"
    assert uberllm["default_model_placeholder"] == "uberllm/deepseek/deepseek-v4.1-flash"
    assert {field["key"]: field["required"] for field in uberllm["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_uberllm_supported_endpoints():
    matrix_path = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    providers = json.loads(matrix_path.read_text())["providers"]

    assert providers["uberllm"]["endpoints"] == {
        "chat_completions": True,
        "messages": True,
        "responses": False,
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


def test_uberllm_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.uberllm.dev/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_uberllm",
                "object": "chat.completion",
                "created": 1_793_330_000,
                "model": "deepseek/deepseek-v4.1-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from UberLLM"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.completion(
            model="uberllm/deepseek/deepseek-v4.1-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="uberllm-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.uberllm.dev/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer uberllm-test-key"
    assert body["model"] == "deepseek/deepseek-v4.1-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from UberLLM"


@pytest.mark.asyncio
async def test_uberllm_anthropic_messages_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.uberllm.dev/v1/messages").respond(
            200,
            json={
                "id": "msg_uberllm",
                "type": "message",
                "role": "assistant",
                "model": "deepseek/deepseek-v4.1-flash",
                "content": [{"type": "text", "text": "Hello from UberLLM"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="uberllm/deepseek/deepseek-v4.1-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="uberllm-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.uberllm.dev/v1/messages"
    assert request.headers["authorization"] == "Bearer uberllm-test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "deepseek/deepseek-v4.1-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from UberLLM"
