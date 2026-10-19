import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_oceantoken",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "deepseek/deepseek-v4-pro",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from OceanToken"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}


def test_oceantoken_is_a_registered_provider():
    assert litellm.LlmProviders.OCEANTOKEN.value == "oceantoken"
    assert "oceantoken" in litellm.provider_list
    assert "oceantoken" in litellm.constants.openai_compatible_providers


def test_oceantoken_provider_resolution_keeps_publisher_prefix(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("OCEANTOKEN_API_KEY", "oceantoken-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="oceantoken/deepseek/deepseek-v4-pro",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "deepseek/deepseek-v4-pro"
    assert provider == "oceantoken"
    assert api_key == "oceantoken-test-key"
    assert api_base == "https://api.oceantoken.ai/v1"


def test_oceantoken_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("OCEANTOKEN_API_KEY", "oceantoken-env-key")
    monkeypatch.setenv("OCEANTOKEN_API_BASE", "https://oceantoken.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="oceantoken/deepseek/deepseek-v4-pro",
        custom_llm_provider=None,
        api_base="https://oceantoken.internal.example/v1",
        api_key="oceantoken-explicit-key",
    )

    assert provider == "oceantoken"
    assert api_key == "oceantoken-explicit-key"
    assert api_base == "https://oceantoken.internal.example/v1"


def test_oceantoken_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("OCEANTOKEN_API_KEY", "oceantoken-env-key")
    monkeypatch.setenv("OCEANTOKEN_API_BASE", "https://oceantoken.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="oceantoken/deepseek/deepseek-v4-pro",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "oceantoken"
    assert api_base == "https://oceantoken.env.example/v1"


def test_oceantoken_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    oceantoken = next(provider for provider in providers if provider["litellm_provider"] == "oceantoken")

    assert oceantoken["provider"] == "OCEANTOKEN"
    assert oceantoken["provider_display_name"] == "OceanToken"
    assert oceantoken["default_model_placeholder"] == "oceantoken/deepseek/deepseek-v4-pro"
    assert {field["key"]: field["required"] for field in oceantoken["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_oceantoken_supported_endpoints():
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

    assert json.loads(backup_path.read_text())["providers"]["oceantoken"]["endpoints"] == expected
    assert json.loads(root_path.read_text())["providers"]["oceantoken"]["endpoints"] == expected


def test_oceantoken_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.oceantoken.ai/v1/chat/completions").respond(
            200, json=_CHAT_COMPLETION
        )
        response: Final = litellm.completion(
            model="oceantoken/deepseek/deepseek-v4-pro",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="oceantoken-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.oceantoken.ai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer oceantoken-test-key"
    assert body["model"] == "deepseek/deepseek-v4-pro"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from OceanToken"


def test_oceantoken_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.oceantoken.ai/v1/responses").respond(
            200,
            json={
                "id": "resp_oceantoken",
                "object": "response",
                "created_at": 1_790_000_000,
                "model": "openai/gpt-5.5",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_oceantoken",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from OceanToken", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="oceantoken/openai/gpt-5.5",
            input="Say hello",
            api_key="oceantoken-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.oceantoken.ai/v1/responses"
    assert request.headers["authorization"] == "Bearer oceantoken-test-key"
    assert body["model"] == "openai/gpt-5.5"
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from OceanToken"


@pytest.mark.asyncio
async def test_oceantoken_anthropic_messages_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.oceantoken.ai/v1/messages").respond(
            200,
            json={
                "id": "msg_oceantoken",
                "type": "message",
                "role": "assistant",
                "model": "anthropic/claude-sonnet-5.5",
                "content": [{"type": "text", "text": "Hello from OceanToken"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="oceantoken/anthropic/claude-sonnet-5.5",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="oceantoken-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.oceantoken.ai/v1/messages"
    assert request.headers["authorization"] == "Bearer oceantoken-test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "anthropic/claude-sonnet-5.5"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from OceanToken"
