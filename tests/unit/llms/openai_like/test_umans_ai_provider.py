import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache


def test_umans_ai_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("UMANS_AI_API_KEY", "umans-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="umans-ai/umans-deepseek-v4-flash-0731",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "umans-deepseek-v4-flash-0731"
    assert provider == "umans-ai"
    assert api_key == "umans-test-key"
    assert api_base == "https://api.code.umans.ai/v1"


def test_umans_ai_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("UMANS_AI_API_KEY", "umans-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="umans-ai/umans-deepseek-v4-flash-0731",
        custom_llm_provider=None,
        api_base="https://umans.internal.example/v1",
        api_key="umans-explicit-key",
    )

    assert provider == "umans-ai"
    assert api_key == "umans-explicit-key"
    assert api_base == "https://umans.internal.example/v1"


def test_umans_ai_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    umans = next(provider for provider in providers if provider["litellm_provider"] == "umans-ai")

    assert umans["provider"] == "UMANS_AI"
    assert umans["provider_display_name"] == "Umans AI"
    assert umans["default_model_placeholder"] == "umans-ai/umans-deepseek-v4-flash-0731"
    assert {field["key"]: field["required"] for field in umans["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_umans_ai_supported_endpoints():
    matrix_path = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    providers = json.loads(matrix_path.read_text())["providers"]

    assert providers["umans-ai"]["endpoints"] == {
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


def test_umans_ai_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.code.umans.ai/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_umans",
                "object": "chat.completion",
                "created": 1_789_550_000,
                "model": "umans-deepseek-v4-flash-0731",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from Umans AI"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.completion(
            model="umans-ai/umans-deepseek-v4-flash-0731",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="umans-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.code.umans.ai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer umans-test-key"
    assert body["model"] == "umans-deepseek-v4-flash-0731"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Umans AI"


def test_umans_ai_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.code.umans.ai/v1/responses").respond(
            200,
            json={
                "id": "resp_umans",
                "object": "response",
                "created_at": 1_789_550_000,
                "model": "umans-deepseek-v4-flash-0731",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_umans",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from Umans AI", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="umans-ai/umans-deepseek-v4-flash-0731",
            input="Say hello",
            api_key="umans-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.code.umans.ai/v1/responses"
    assert request.headers["authorization"] == "Bearer umans-test-key"
    assert body["model"] == "umans-deepseek-v4-flash-0731"
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from Umans AI"


@pytest.mark.asyncio
async def test_umans_ai_anthropic_messages_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.code.umans.ai/v1/messages").respond(
            200,
            json={
                "id": "msg_umans",
                "type": "message",
                "role": "assistant",
                "model": "umans-deepseek-v4-flash-0731",
                "content": [{"type": "text", "text": "Hello from Umans AI"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="umans-ai/umans-deepseek-v4-flash-0731",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="umans-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.code.umans.ai/v1/messages"
    assert request.headers["authorization"] == "Bearer umans-test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "umans-deepseek-v4-flash-0731"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from Umans AI"
