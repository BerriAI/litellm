import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_nearai",
    "object": "chat.completion",
    "created": 1_792_000_000,
    "model": "z-ai/glm-5.3-flash",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from NEAR AI"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8},
}


def test_nearai_is_a_registered_provider():
    assert litellm.LlmProviders.NEARAI.value == "nearai"
    assert "nearai" in litellm.provider_list
    assert "nearai" in litellm.constants.openai_compatible_providers


def test_nearai_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("NEARAI_API_KEY", "nearai-test-key")
    monkeypatch.delenv("NEARAI_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="nearai/z-ai/glm-5.3-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "z-ai/glm-5.3-flash"
    assert provider == "nearai"
    assert api_key == "nearai-test-key"
    assert api_base == "https://cloud-api.near.ai/v1"


def test_nearai_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("NEARAI_API_KEY", "nearai-env-key")
    monkeypatch.setenv("NEARAI_API_BASE", "https://nearai.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="nearai/z-ai/glm-5.3-flash",
        custom_llm_provider=None,
        api_base="https://nearai.internal.example/v1",
        api_key="nearai-explicit-key",
    )

    assert provider == "nearai"
    assert api_key == "nearai-explicit-key"
    assert api_base == "https://nearai.internal.example/v1"


def test_nearai_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("NEARAI_API_KEY", "nearai-env-key")
    monkeypatch.setenv("NEARAI_API_BASE", "https://nearai.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="nearai/z-ai/glm-5.3-flash",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "nearai"
    assert api_base == "https://nearai.env.example/v1"


def test_nearai_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("NEARAI_API_KEY", "nearai-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="z-ai/glm-5.3-flash",
        custom_llm_provider=None,
        api_base="https://cloud-api.near.ai/v1",
        api_key=None,
    )

    assert model == "z-ai/glm-5.3-flash"
    assert provider == "nearai"
    assert api_key == "nearai-env-key"
    assert api_base == "https://cloud-api.near.ai/v1"


def test_nearai_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    nearai = next(provider for provider in providers if provider["litellm_provider"] == "nearai")

    assert nearai["provider"] == "NEARAI"
    assert nearai["provider_display_name"] == "NEAR AI Cloud"
    assert nearai["default_model_placeholder"] == "nearai/z-ai/glm-5.3-flash"
    assert {field["key"]: field["required"] for field in nearai["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_nearai_supported_endpoints():
    expected: Final = {
        "chat_completions": True,
        "messages": True,
        "responses": True,
        "embeddings": True,
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

    assert json.loads(backup_path.read_text())["providers"]["nearai"]["endpoints"] == expected
    assert json.loads(root_path.read_text())["providers"]["nearai"]["endpoints"] == expected


def test_nearai_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://cloud-api.near.ai/v1/chat/completions").respond(
            200, json=_CHAT_COMPLETION
        )
        response: Final = litellm.completion(
            model="nearai/z-ai/glm-5.3-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=32,
            api_key="nearai-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://cloud-api.near.ai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer nearai-test-key"
    assert body["model"] == "z-ai/glm-5.3-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_completion_tokens"] == 32
    assert response.choices[0].message.content == "Hello from NEAR AI"


def test_nearai_responses_request_is_bridged_to_chat_completions():
    with respx.mock(assert_all_called=False) as upstream:
        chat_route: Final = upstream.post("https://cloud-api.near.ai/v1/chat/completions").respond(
            200, json=_CHAT_COMPLETION
        )
        responses_route: Final = upstream.post("https://cloud-api.near.ai/v1/responses").respond(500)
        response: Final = litellm.responses(
            model="nearai/z-ai/glm-5.3-flash",
            input="Say hello",
            api_key="nearai-test-key",
        )

    request: Final = chat_route.calls.last.request
    body: Final = json.loads(request.content)
    assert chat_route.call_count == 1
    assert responses_route.call_count == 0
    assert request.headers["authorization"] == "Bearer nearai-test-key"
    assert body["model"] == "z-ai/glm-5.3-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.output[0].content[0].text == "Hello from NEAR AI"


@pytest.mark.asyncio
async def test_nearai_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock(assert_all_called=False) as upstream:
        chat_route: Final = upstream.post("https://cloud-api.near.ai/v1/chat/completions").respond(
            200, json=_CHAT_COMPLETION
        )
        messages_route: Final = upstream.post("https://cloud-api.near.ai/v1/messages").respond(500)
        response: Final = await litellm.anthropic.messages.acreate(
            model="nearai/z-ai/glm-5.3-flash",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="nearai-test-key",
        )

    request: Final = chat_route.calls.last.request
    body: Final = json.loads(request.content)
    assert chat_route.call_count == 1
    assert messages_route.call_count == 0
    assert request.headers["authorization"] == "Bearer nearai-test-key"
    assert body["model"] == "z-ai/glm-5.3-flash"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from NEAR AI"


def test_nearai_embedding_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://cloud-api.near.ai/v1/embeddings").respond(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.25, -0.5, 0.75]}],
                "model": "Qwen/Qwen3-Embedding-0.6B",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )
        response: Final = litellm.embedding(
            model="nearai/Qwen/Qwen3-Embedding-0.6B",
            input=["Say hello"],
            api_key="nearai-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://cloud-api.near.ai/v1/embeddings"
    assert request.headers["authorization"] == "Bearer nearai-test-key"
    assert body["model"] == "Qwen/Qwen3-Embedding-0.6B"
    assert body["input"] == ["Say hello"]
    assert response.data[0]["embedding"] == [0.25, -0.5, 0.75]
