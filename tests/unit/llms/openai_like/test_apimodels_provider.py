import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache


def _chat_completion(model: str) -> dict:
    return {
        "id": "chatcmpl_apimodels",
        "object": "chat.completion",
        "created": 1_790_000_000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from APIMODELS"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
    }


def test_apimodels_is_a_registered_provider():
    assert litellm.LlmProviders.APIMODELS.value == "apimodels"
    assert "apimodels" in litellm.provider_list
    assert "apimodels" in litellm.constants.openai_compatible_providers


def test_apimodels_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("APIMODELS_API_KEY", "apimodels-test-key")
    monkeypatch.delenv("APIMODELS_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="apimodels/gpt-5.6-luna",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "gpt-5.6-luna"
    assert provider == "apimodels"
    assert api_key == "apimodels-test-key"
    assert api_base == "https://api.apimodels.app/v1"


def test_apimodels_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("APIMODELS_API_KEY", "apimodels-env-key")
    monkeypatch.setenv("APIMODELS_API_BASE", "https://apimodels.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="apimodels/gpt-5.6-luna",
        custom_llm_provider=None,
        api_base="https://apimodels.internal.example/v1",
        api_key="apimodels-explicit-key",
    )

    assert provider == "apimodels"
    assert api_key == "apimodels-explicit-key"
    assert api_base == "https://apimodels.internal.example/v1"


def test_apimodels_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("APIMODELS_API_KEY", "apimodels-env-key")
    monkeypatch.setenv("APIMODELS_API_BASE", "https://apimodels.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="apimodels/gpt-5.6-luna",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "apimodels"
    assert api_base == "https://apimodels.env.example/v1"


def test_apimodels_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("APIMODELS_API_KEY", "apimodels-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="gpt-5.6-luna",
        custom_llm_provider=None,
        api_base="https://api.apimodels.app/v1",
        api_key=None,
    )

    assert model == "gpt-5.6-luna"
    assert provider == "apimodels"
    assert api_key == "apimodels-env-key"
    assert api_base == "https://api.apimodels.app/v1"


def test_apimodels_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    apimodels = next(provider for provider in providers if provider["litellm_provider"] == "apimodels")

    assert apimodels["provider"] == "APIMODELS"
    assert apimodels["provider_display_name"] == "apimodels.app"
    assert apimodels["default_model_placeholder"] == "apimodels/claude-opus-5-5"
    assert {field["key"]: field["required"] for field in apimodels["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_apimodels_supported_endpoints():
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

    assert json.loads(backup_path.read_text())["providers"]["apimodels"]["endpoints"] == expected
    assert json.loads(root_path.read_text())["providers"]["apimodels"]["endpoints"] == expected


def test_apimodels_chat_completion_request_uses_env_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APIMODELS_API_KEY", "apimodels-env-key")
    monkeypatch.delenv("APIMODELS_API_BASE", raising=False)
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.apimodels.app/v1/chat/completions").respond(
            200, json=_chat_completion("gpt-5.6-luna")
        )
        response: Final = litellm.completion(
            model="apimodels/gpt-5.6-luna",
            messages=[{"role": "user", "content": "Say hello"}],
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.apimodels.app/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer apimodels-env-key"
    assert body["model"] == "gpt-5.6-luna"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from APIMODELS"


def test_apimodels_chat_completion_request_honours_api_base_override():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://apimodels.internal.example/v1/chat/completions").respond(
            200, json=_chat_completion("gpt-5.6-luna")
        )
        response: Final = litellm.completion(
            model="apimodels/gpt-5.6-luna",
            messages=[{"role": "user", "content": "Say hello"}],
            api_base="https://apimodels.internal.example/v1",
            api_key="apimodels-explicit-key",
        )

    request: Final = route.calls.last.request
    assert route.call_count == 1
    assert str(request.url) == "https://apimodels.internal.example/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer apimodels-explicit-key"
    assert response.choices[0].message.content == "Hello from APIMODELS"


def test_apimodels_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.apimodels.app/v1/chat/completions").respond(
            200, json=_chat_completion("claude-opus-5-5")
        )
        response: Final = litellm.responses(
            model="apimodels/claude-opus-5-5",
            input="Say hello",
            api_key="apimodels-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer apimodels-test-key"
    assert body["model"] == "claude-opus-5-5"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.output[0].content[0].text == "Hello from APIMODELS"


@pytest.mark.asyncio
async def test_apimodels_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.apimodels.app/v1/chat/completions").respond(
            200, json=_chat_completion("gpt-5.6-luna")
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="apimodels/gpt-5.6-luna",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="apimodels-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer apimodels-test-key"
    assert body["model"] == "gpt-5.6-luna"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert response["content"][0]["text"] == "Hello from APIMODELS"
