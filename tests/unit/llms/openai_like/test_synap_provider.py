import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache


def test_synap_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("SYNAP_API_KEY", "synap-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="synap/qwen/qwen3-coder-30b-a3b-instruct",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "qwen/qwen3-coder-30b-a3b-instruct"
    assert provider == "synap"
    assert api_key == "synap-test-key"
    assert api_base == "https://pool.linkrra.com/v1"


def test_synap_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("SYNAP_API_KEY", "synap-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model="synap/qwen/qwen3-coder-30b-a3b-instruct",
        custom_llm_provider=None,
        api_base="https://synap.internal.example/v1",
        api_key="synap-explicit-key",
    )

    assert provider == "synap"
    assert api_key == "synap-explicit-key"
    assert api_base == "https://synap.internal.example/v1"


def test_synap_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    synap: Final = next(provider for provider in providers if provider["litellm_provider"] == "synap")

    assert synap["provider"] == "SYNAP"
    assert synap["provider_display_name"] == "Synap"
    assert synap["default_model_placeholder"] == "synap/qwen/qwen3-coder-30b-a3b-instruct"
    assert {field["key"]: field["required"] for field in synap["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_synap_supported_endpoints():
    matrix_path: Final = Path(litellm.__file__).parent / "provider_endpoints_support_backup.json"
    providers: Final = json.loads(matrix_path.read_text())["providers"]

    assert providers["synap"]["endpoints"] == {
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


def test_synap_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://pool.linkrra.com/v1/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_synap",
                "object": "chat.completion",
                "created": 1_791_190_000,
                "model": "qwen/qwen3-coder-30b-a3b-instruct",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from Synap"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.completion(
            model="synap/qwen/qwen3-coder-30b-a3b-instruct",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="synap-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://pool.linkrra.com/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer synap-test-key"
    assert body["model"] == "qwen/qwen3-coder-30b-a3b-instruct"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Synap"


@pytest.mark.asyncio
async def test_synap_anthropic_messages_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://pool.linkrra.com/v1/messages").respond(
            200,
            json={
                "id": "msg_synap",
                "type": "message",
                "role": "assistant",
                "model": "qwen/qwen3-coder-30b-a3b-instruct",
                "content": [{"type": "text", "text": "Hello from Synap"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="synap/qwen/qwen3-coder-30b-a3b-instruct",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="synap-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://pool.linkrra.com/v1/messages"
    assert request.headers["authorization"] == "Bearer synap-test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "qwen/qwen3-coder-30b-a3b-instruct"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from Synap"


def test_synap_completion_cost_is_calculated_from_usage():
    """Spend for a Synap call is computed by completion_cost(), not read from the map.

    1M prompt + 1M completion tokens makes the expectation the published per-million
    price directly: $0.0675 in, $0.27 out.
    """
    from litellm import completion_cost
    from litellm.types.utils import Choices, Message, ModelResponse, Usage

    response: Final = ModelResponse(
        id="chatcmpl_synap_cost",
        created=1_791_190_000,
        model="qwen/qwen3-coder-30b-a3b-instruct",
        object="chat.completion",
        choices=[
            Choices(
                index=0,
                message=Message(role="assistant", content="priced"),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000),
    )

    cost: Final = completion_cost(
        completion_response=response,
        model="synap/qwen/qwen3-coder-30b-a3b-instruct",
        custom_llm_provider="synap",
    )

    assert cost == pytest.approx(0.0675 + 0.27)
