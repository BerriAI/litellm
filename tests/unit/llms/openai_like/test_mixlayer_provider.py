import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_mixlayer",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "z-ai/glm-5.3",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Mixlayer"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}

_REPO_ROOT: Final = Path(litellm.__file__).parent.parent

# USD per token. Published rates are USD per million tokens. Max output tokens
# are not published, so the cost-map rows omit max_output_tokens and max_tokens.
_MODELS: Final = {
    "mixlayer/z-ai/glm-5.3": {
        "input_cost_per_token": 1.15e-6,
        "output_cost_per_token": 3.9e-6,
        "max_input_tokens": 262144,
        "supports_vision": False,
        "thinking_always_on": True,
    },
    "mixlayer/qwen/qwen3.8-27b": {
        "input_cost_per_token": 2.5e-7,
        "output_cost_per_token": 1.95e-6,
        "max_input_tokens": 262144,
        "supports_vision": True,
    },
    "mixlayer/qwen/qwen3.6-35b-a3b": {
        "input_cost_per_token": 2.5e-7,
        "output_cost_per_token": 1.3e-6,
        "max_input_tokens": 131072,
        "supports_vision": True,
    },
    "mixlayer/deepseek/deepseek-v4.1-flash": {
        "input_cost_per_token": 3e-7,
        "output_cost_per_token": 1.2e-6,
        "max_input_tokens": 262144,
        "supports_vision": False,
    },
    "mixlayer/qwen/qwen3.5-9b": {
        "input_cost_per_token": 1e-7,
        "output_cost_per_token": 4e-7,
        "max_input_tokens": 131072,
        "supports_vision": True,
    },
    "mixlayer/qwen/qwen3.5-4b-free": {
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "max_input_tokens": 131072,
        "supports_vision": True,
    },
}


def test_mixlayer_is_a_registered_provider():
    assert litellm.LlmProviders.MIXLAYER.value == "mixlayer"
    assert "mixlayer" in litellm.provider_list
    assert "mixlayer" in litellm.constants.openai_compatible_providers
    assert "https://mixlayer.ai/v1" in litellm.constants.openai_compatible_endpoints


def test_mixlayer_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("MIXLAYER_API_KEY", "mixlayer-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="mixlayer/z-ai/glm-5.3",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "z-ai/glm-5.3"
    assert provider == "mixlayer"
    assert api_key == "mixlayer-test-key"
    assert api_base == "https://mixlayer.ai/v1"


def test_mixlayer_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("MIXLAYER_API_KEY", "mixlayer-env-key")
    monkeypatch.setenv("MIXLAYER_API_BASE", "https://mixlayer.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="mixlayer/z-ai/glm-5.3",
        custom_llm_provider=None,
        api_base="https://mixlayer.internal.example/v1",
        api_key="mixlayer-explicit-key",
    )

    assert provider == "mixlayer"
    assert api_key == "mixlayer-explicit-key"
    assert api_base == "https://mixlayer.internal.example/v1"


def test_mixlayer_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("MIXLAYER_API_KEY", "mixlayer-env-key")
    monkeypatch.setenv("MIXLAYER_API_BASE", "https://mixlayer.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="mixlayer/z-ai/glm-5.3",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "mixlayer"
    assert api_base == "https://mixlayer.env.example/v1"


def test_mixlayer_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("MIXLAYER_API_KEY", "mixlayer-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="z-ai/glm-5.3",
        custom_llm_provider=None,
        api_base="https://mixlayer.ai/v1",
        api_key=None,
    )

    assert model == "z-ai/glm-5.3"
    assert provider == "mixlayer"
    assert api_key == "mixlayer-env-key"
    assert api_base == "https://mixlayer.ai/v1"


def test_mixlayer_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    mixlayer = next(provider for provider in providers if provider["litellm_provider"] == "mixlayer")

    assert mixlayer["provider"] == "MIXLAYER"
    assert mixlayer["provider_display_name"] == "Mixlayer"
    assert mixlayer["default_model_placeholder"] == "mixlayer/z-ai/glm-5.3"
    assert {field["key"]: field["required"] for field in mixlayer["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_mixlayer_supported_endpoints():
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
    root_path = _REPO_ROOT / "provider_endpoints_support.json"

    assert json.loads(backup_path.read_text())["providers"]["mixlayer"]["endpoints"] == expected
    assert json.loads(root_path.read_text())["providers"]["mixlayer"]["endpoints"] == expected


def test_mixlayer_cost_map_rows():
    prices_path = _REPO_ROOT / "model_prices_and_context_window.json"
    backup_path = Path(litellm.__file__).parent / "model_prices_and_context_window_backup.json"
    prices: Final = json.loads(prices_path.read_text())
    backup: Final = json.loads(backup_path.read_text())

    for model, expected in _MODELS.items():
        for catalog in (prices, backup):
            row = catalog[model]
            assert row["litellm_provider"] == "mixlayer"
            assert row["mode"] == "chat"
            assert row["source"] == "https://mixlayer.com/pricing"
            assert row["supported_endpoints"] == ["/v1/chat/completions", "/v1/responses"]
            assert row["supports_function_calling"] is True
            assert row["supports_reasoning"] is True
            assert row["supports_response_schema"] is True
            assert row["supports_native_streaming"] is True
            assert "supports_tool_choice" not in row
            assert "max_output_tokens" not in row
            assert "max_tokens" not in row
            assert "cache_read_input_token_cost" not in row
            for field, value in expected.items():
                assert row[field] == value


def test_mixlayer_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://mixlayer.ai/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="mixlayer/z-ai/glm-5.3",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="mixlayer-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://mixlayer.ai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer mixlayer-test-key"
    assert body["model"] == "z-ai/glm-5.3"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Mixlayer"


def test_mixlayer_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://mixlayer.ai/v1/responses").respond(
            200,
            json={
                "id": "resp_mixlayer",
                "object": "response",
                "created_at": 1_789_550_000,
                "model": "z-ai/glm-5.3",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_mixlayer",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from Mixlayer", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model="mixlayer/z-ai/glm-5.3",
            input="Say hello",
            api_key="mixlayer-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://mixlayer.ai/v1/responses"
    assert request.headers["authorization"] == "Bearer mixlayer-test-key"
    assert body["model"] == "z-ai/glm-5.3"
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from Mixlayer"


@pytest.mark.asyncio
async def test_mixlayer_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://mixlayer.ai/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model="mixlayer/z-ai/glm-5.3",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="mixlayer-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer mixlayer-test-key"
    assert body["model"] == "z-ai/glm-5.3"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert response["content"][0]["text"] == "Hello from Mixlayer"
