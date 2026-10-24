import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache


def test_inferall_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("INFERALL_API_KEY", "inferall-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="inferall/openai/gpt-oss-20b",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "openai/gpt-oss-20b"
    assert provider == "inferall"
    assert api_key == "inferall-test-key"
    assert api_base == "https://api.inferall.ai/v1"


def test_inferall_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("INFERALL_API_KEY", "inferall-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="inferall/openai/gpt-oss-20b",
        custom_llm_provider=None,
        api_base="https://inferall.internal.example/v1",
        api_key="inferall-explicit-key",
    )

    assert provider == "inferall"
    assert api_key == "inferall-explicit-key"
    assert api_base == "https://inferall.internal.example/v1"


def test_inferall_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    inferall = next(provider for provider in providers if provider["litellm_provider"] == "inferall")

    assert inferall["provider"] == "INFERALL"
    assert inferall["provider_display_name"] == "InferAll"
    assert inferall["default_model_placeholder"] == "inferall/nvidia/nemotron-3-ultra-550b-a55b"
    assert {field["key"]: field["required"] for field in inferall["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_inferall_supported_endpoints():
    matrix_path = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    providers = json.loads(matrix_path.read_text())["providers"]

    assert providers["inferall"]["endpoints"] == {
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


def test_inferall_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.inferall.ai/v1/responses").respond(
            200,
            json={
                "id": "resp_inferall",
                "object": "response",
                "created_at": 1_789_550_000,
                "model": "gpt-oss-20b",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_inferall",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from InferAll", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="inferall/openai/gpt-oss-20b",
            input="Say hello",
            api_key="inferall-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.inferall.ai/v1/responses"
    assert request.headers["authorization"] == "Bearer inferall-test-key"
    assert body["model"] == "openai/gpt-oss-20b"
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from InferAll"


@pytest.mark.asyncio
async def test_inferall_anthropic_messages_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.inferall.ai/v1/messages").respond(
            200,
            json={
                "id": "msg_inferall",
                "type": "message",
                "role": "assistant",
                "model": "gpt-oss-20b",
                "content": [{"type": "text", "text": "Hello from InferAll"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="inferall/openai/gpt-oss-20b",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="inferall-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.inferall.ai/v1/messages"
    assert request.headers["authorization"] == "Bearer inferall-test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "openai/gpt-oss-20b"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from InferAll"


def test_inferall_declares_tools_and_reasoning_for_unmapped_models():
    from litellm.utils import get_supported_openai_params

    params: Final = get_supported_openai_params(model="openai/gpt-oss-20b", custom_llm_provider="inferall")

    assert params is not None
    assert "tools" in params
    assert "tool_choice" in params
    assert "reasoning_effort" in params
