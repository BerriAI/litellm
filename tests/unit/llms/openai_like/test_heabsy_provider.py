import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_heabsy",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "qwen38",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Heabsy"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}


def test_heabsy_is_a_registered_provider():
    assert litellm.LlmProviders.HEABSY.value == "heabsy"
    assert "heabsy" in litellm.provider_list
    assert "heabsy" in litellm.constants.openai_compatible_providers


def test_heabsy_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("HEABSY_API_KEY", "heabsy-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="heabsy/qwen38",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "qwen38"
    assert provider == "heabsy"
    assert api_key == "heabsy-test-key"
    assert api_base == "https://api.heabsy.com/v1"


def test_heabsy_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("HEABSY_API_KEY", "heabsy-env-key")
    monkeypatch.setenv("HEABSY_API_BASE", "https://heabsy.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="heabsy/qwen38",
        custom_llm_provider=None,
        api_base="https://heabsy.internal.example/v1",
        api_key="heabsy-explicit-key",
    )

    assert provider == "heabsy"
    assert api_key == "heabsy-explicit-key"
    assert api_base == "https://heabsy.internal.example/v1"


def test_heabsy_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("HEABSY_API_KEY", "heabsy-env-key")
    monkeypatch.setenv("HEABSY_API_BASE", "https://heabsy.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="heabsy/qwen38",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "heabsy"
    assert api_base == "https://heabsy.env.example/v1"


def test_heabsy_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("HEABSY_API_KEY", "heabsy-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="qwen38",
        custom_llm_provider=None,
        api_base="https://api.heabsy.com/v1",
        api_key=None,
    )

    assert model == "qwen38"
    assert provider == "heabsy"
    assert api_key == "heabsy-env-key"
    assert api_base == "https://api.heabsy.com/v1"


def test_heabsy_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    heabsy = next(provider for provider in providers if provider["litellm_provider"] == "heabsy")

    assert heabsy["provider"] == "HEABSY"
    assert heabsy["provider_display_name"] == "Heabsy"
    assert heabsy["default_model_placeholder"] == "heabsy/qwen38"
    assert {field["key"]: field["required"] for field in heabsy["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_heabsy_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.heabsy.com/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="heabsy/qwen38",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="heabsy-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.heabsy.com/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer heabsy-test-key"
    assert body["model"] == "qwen38"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Heabsy"


def test_heabsy_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.heabsy.com/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="heabsy/qwen38",
            input="Say hello",
            api_key="heabsy-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer heabsy-test-key"
    assert body["model"] == "qwen38"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.output[0].content[0].text == "Hello from Heabsy"


@pytest.mark.asyncio
async def test_heabsy_anthropic_messages_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.heabsy.com/v1/messages").respond(
            200,
            json={
                "id": "msg_heabsy",
                "type": "message",
                "role": "assistant",
                "model": "qwen38",
                "content": [{"type": "text", "text": "Hello from Heabsy"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="heabsy/qwen38",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="heabsy-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.heabsy.com/v1/messages"
    assert request.headers["authorization"] == "Bearer heabsy-test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "qwen38"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from Heabsy"
