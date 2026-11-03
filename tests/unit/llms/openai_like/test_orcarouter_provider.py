import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_orcarouter",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "openai/gpt-4.1-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from OrcaRouter"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}


def test_orcarouter_is_a_registered_provider():
    assert litellm.LlmProviders.ORCAROUTER.value == "orcarouter"
    assert "orcarouter" in litellm.provider_list
    assert "orcarouter" in litellm.constants.openai_compatible_providers


def test_orcarouter_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("ORCAROUTER_API_KEY", "orcarouter-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="orcarouter/openai/gpt-4.1-mini",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "openai/gpt-4.1-mini"
    assert provider == "orcarouter"
    assert api_key == "orcarouter-test-key"
    assert api_base == "https://api.orcarouter.ai/v1"


def test_orcarouter_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("ORCAROUTER_API_KEY", "orcarouter-env-key")
    monkeypatch.setenv("ORCAROUTER_API_BASE", "https://orcarouter.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="orcarouter/openai/gpt-4.1-mini",
        custom_llm_provider=None,
        api_base="https://orcarouter.internal.example/v1",
        api_key="orcarouter-explicit-key",
    )

    assert provider == "orcarouter"
    assert api_key == "orcarouter-explicit-key"
    assert api_base == "https://orcarouter.internal.example/v1"


def test_orcarouter_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("ORCAROUTER_API_KEY", "orcarouter-env-key")
    monkeypatch.setenv("ORCAROUTER_API_BASE", "https://orcarouter.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="orcarouter/openai/gpt-4.1-mini",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "orcarouter"
    assert api_base == "https://orcarouter.env.example/v1"


def test_orcarouter_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("ORCAROUTER_API_KEY", "orcarouter-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="gpt-4.1-mini",
        custom_llm_provider=None,
        api_base="https://api.orcarouter.ai/v1",
        api_key=None,
    )

    assert model == "gpt-4.1-mini"
    assert provider == "orcarouter"
    assert api_key == "orcarouter-env-key"
    assert api_base == "https://api.orcarouter.ai/v1"


def test_orcarouter_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    orcarouter = next(provider for provider in providers if provider["litellm_provider"] == "orcarouter")

    assert orcarouter["provider"] == "ORCAROUTER"
    assert orcarouter["provider_display_name"] == "OrcaRouter"
    assert orcarouter["default_model_placeholder"] == "orcarouter/openai/gpt-4.1-mini"
    assert {field["key"]: field["required"] for field in orcarouter["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_orcarouter_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.orcarouter.ai/v1/chat/completions").respond(
            200, json=_CHAT_COMPLETION
        )
        response: Final = litellm.completion(
            model="orcarouter/openai/gpt-4.1-mini",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="orcarouter-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.orcarouter.ai/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer orcarouter-test-key"
    assert body["model"] == "openai/gpt-4.1-mini"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from OrcaRouter"


def test_orcarouter_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.orcarouter.ai/v1/chat/completions").respond(
            200, json=_CHAT_COMPLETION
        )
        response: Final = litellm.responses(
            model="orcarouter/openai/gpt-4.1-mini",
            input="Say hello",
            api_key="orcarouter-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer orcarouter-test-key"
    assert body["model"] == "openai/gpt-4.1-mini"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.output[0].content[0].text == "Hello from OrcaRouter"


@pytest.mark.asyncio
async def test_orcarouter_anthropic_messages_request_is_bridged_to_chat_completions(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.orcarouter.ai/v1/chat/completions").respond(
            200, json=_CHAT_COMPLETION
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="orcarouter/openai/gpt-4.1-mini",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="orcarouter-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer orcarouter-test-key"
    assert body["model"] == "openai/gpt-4.1-mini"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert response["content"][0]["text"] == "Hello from OrcaRouter"
