import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_rapidrouter",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "anthropic/claude-sonnet-5",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Rapid Router"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}


def test_rapidrouter_is_a_registered_provider():
    assert litellm.LlmProviders.RAPIDROUTER.value == "rapidrouter"
    assert "rapidrouter" in litellm.provider_list
    assert "rapidrouter" in litellm.constants.openai_compatible_providers


def test_rapidrouter_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("RAPIDROUTER_API_KEY", "rapidrouter-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="rapidrouter/anthropic/claude-sonnet-5",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "anthropic/claude-sonnet-5"
    assert provider == "rapidrouter"
    assert api_key == "rapidrouter-test-key"
    assert api_base == "https://rapidrouter.ai/api/v1"


def test_rapidrouter_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("RAPIDROUTER_API_KEY", "rapidrouter-env-key")
    monkeypatch.setenv("RAPIDROUTER_API_BASE", "https://rapidrouter.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="rapidrouter/anthropic/claude-sonnet-5",
        custom_llm_provider=None,
        api_base="https://rapidrouter.internal.example/v1",
        api_key="rapidrouter-explicit-key",
    )

    assert provider == "rapidrouter"
    assert api_key == "rapidrouter-explicit-key"
    assert api_base == "https://rapidrouter.internal.example/v1"


def test_rapidrouter_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("RAPIDROUTER_API_KEY", "rapidrouter-env-key")
    monkeypatch.setenv("RAPIDROUTER_API_BASE", "https://rapidrouter.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="rapidrouter/anthropic/claude-sonnet-5",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "rapidrouter"
    assert api_base == "https://rapidrouter.env.example/v1"


def test_rapidrouter_keeps_the_author_segment_of_the_model_id(monkeypatch: pytest.MonkeyPatch):
    """Rapid Router model ids are author/model, so only the rapidrouter/ route prefix is stripped."""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("RAPIDROUTER_API_KEY", "rapidrouter-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="rapidrouter/deepseek/deepseek-v4-pro",
        custom_llm_provider=None,
        api_base="https://rapidrouter.ai/api/v1",
        api_key=None,
    )

    assert model == "deepseek/deepseek-v4-pro"
    assert provider == "rapidrouter"
    assert api_key == "rapidrouter-env-key"
    assert api_base == "https://rapidrouter.ai/api/v1"


def test_rapidrouter_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    rapidrouter = next(provider for provider in providers if provider["litellm_provider"] == "rapidrouter")

    assert rapidrouter["provider"] == "RAPIDROUTER"
    assert rapidrouter["provider_display_name"] == "Rapid Router"
    assert rapidrouter["default_model_placeholder"] == "rapidrouter/anthropic/claude-sonnet-5"
    assert {field["key"]: field["required"] for field in rapidrouter["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_rapidrouter_supported_endpoints():
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

    assert json.loads(backup_path.read_text())["providers"]["rapidrouter"]["endpoints"] == expected
    assert json.loads(root_path.read_text())["providers"]["rapidrouter"]["endpoints"] == expected


def test_rapidrouter_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://rapidrouter.ai/api/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="rapidrouter/anthropic/claude-sonnet-5",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="rapidrouter-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://rapidrouter.ai/api/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer rapidrouter-test-key"
    assert body["model"] == "anthropic/claude-sonnet-5"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Rapid Router"


def test_rapidrouter_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://rapidrouter.ai/api/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="rapidrouter/anthropic/claude-sonnet-5",
            input="Say hello",
            api_key="rapidrouter-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer rapidrouter-test-key"
    assert body["model"] == "anthropic/claude-sonnet-5"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.output[0].content[0].text == "Hello from Rapid Router"


@pytest.mark.asyncio
async def test_rapidrouter_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://rapidrouter.ai/api/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model="rapidrouter/anthropic/claude-sonnet-5",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="rapidrouter-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer rapidrouter-test-key"
    assert body["model"] == "anthropic/claude-sonnet-5"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert response["content"][0]["text"] == "Hello from Rapid Router"
