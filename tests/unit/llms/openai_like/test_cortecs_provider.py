import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache


def test_cortecs_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("CORTECS_API_KEY", "cortecs-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="cortecs/gpt-6-sol",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "gpt-6-sol"
    assert provider == "cortecs"
    assert api_key == "cortecs-test-key"
    assert api_base == "https://api.cortecs.ai/v1"


def test_cortecs_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("CORTECS_API_KEY", "cortecs-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="cortecs/gpt-6-sol",
        custom_llm_provider=None,
        api_base="https://cortecs.internal.example/v1",
        api_key="cortecs-explicit-key",
    )

    assert provider == "cortecs"
    assert api_key == "cortecs-explicit-key"
    assert api_base == "https://cortecs.internal.example/v1"


def test_cortecs_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    cortecs = next(provider for provider in providers if provider["litellm_provider"] == "cortecs")

    assert cortecs["provider"] == "CORTECS"
    assert cortecs["provider_display_name"] == "Cortecs"
    assert cortecs["default_model_placeholder"] == "cortecs/gpt-6-sol"
    assert {field["key"]: field["required"] for field in cortecs["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_cortecs_supported_endpoints():
    matrix_path = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    providers = json.loads(matrix_path.read_text())["providers"]

    assert providers["cortecs"]["endpoints"] == {
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


def test_cortecs_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.cortecs.ai/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_cortecs",
                "object": "chat.completion",
                "created": 1_789_550_000,
                "model": "gpt-6-sol",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from Cortecs"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.completion(
            model="cortecs/gpt-6-sol",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="cortecs-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.cortecs.ai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer cortecs-test-key"
    assert body["model"] == "gpt-6-sol"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Cortecs"


def test_cortecs_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.cortecs.ai/v1/responses").respond(
            200,
            json={
                "id": "resp_cortecs",
                "object": "response",
                "created_at": 1_789_550_000,
                "model": "gpt-6-sol",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_cortecs",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from Cortecs", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="cortecs/gpt-6-sol",
            input="Say hello",
            api_key="cortecs-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.cortecs.ai/v1/responses"
    assert request.headers["authorization"] == "Bearer cortecs-test-key"
    assert body["model"] == "gpt-6-sol"
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from Cortecs"


@pytest.mark.asyncio
async def test_cortecs_anthropic_messages_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.cortecs.ai/v1/messages").respond(
            200,
            json={
                "id": "msg_cortecs",
                "type": "message",
                "role": "assistant",
                "model": "gpt-6-sol",
                "content": [{"type": "text", "text": "Hello from Cortecs"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="cortecs/gpt-6-sol",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="cortecs-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.cortecs.ai/v1/messages"
    assert request.headers["authorization"] == "Bearer cortecs-test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "gpt-6-sol"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from Cortecs"
