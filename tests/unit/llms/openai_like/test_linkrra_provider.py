import json
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

_API_BASE: Final = "https://pool.linkrra.com/v1"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_linkrra",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "qwen3-coder-30b",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Linkrra"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}


def test_linkrra_is_a_registered_provider():
    assert litellm.LlmProviders.LINKRRA.value == "linkrra"
    assert "linkrra" in litellm.provider_list
    assert "linkrra" in litellm.constants.openai_compatible_providers


def test_linkrra_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.delenv("LINKRRA_API_BASE", raising=False)
    monkeypatch.setenv("LINKRRA_API_KEY", "linkrra-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="linkrra/qwen3-coder-30b",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "qwen3-coder-30b"
    assert provider == "linkrra"
    assert api_key == "linkrra-test-key"
    assert api_base == _API_BASE


def test_linkrra_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("LINKRRA_API_KEY", "linkrra-env-key")
    monkeypatch.setenv("LINKRRA_API_BASE", "https://linkrra.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="linkrra/qwen3-coder-30b",
        custom_llm_provider=None,
        api_base="https://linkrra.internal.example/v1",
        api_key="linkrra-explicit-key",
    )

    assert provider == "linkrra"
    assert api_key == "linkrra-explicit-key"
    assert api_base == "https://linkrra.internal.example/v1"


def test_linkrra_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("LINKRRA_API_KEY", "linkrra-env-key")
    monkeypatch.setenv("LINKRRA_API_BASE", "https://linkrra.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="linkrra/qwen3-coder-30b",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "linkrra"
    assert api_base == "https://linkrra.env.example/v1"


def test_linkrra_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("LINKRRA_API_KEY", "linkrra-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="qwen3-coder-30b",
        custom_llm_provider=None,
        api_base=_API_BASE,
        api_key=None,
    )

    assert model == "qwen3-coder-30b"
    assert provider == "linkrra"
    assert api_key == "linkrra-env-key"
    assert api_base == _API_BASE


@pytest.mark.asyncio
async def test_linkrra_is_served_by_the_add_model_form_endpoint():
    from litellm.proxy.public_endpoints.public_endpoints import get_provider_fields

    fields: Final = await get_provider_fields()
    linkrra: Final = next(provider for provider in fields if provider.litellm_provider == "linkrra")

    assert linkrra.provider_display_name == "Linkrra"
    assert linkrra.default_model_placeholder == "linkrra/qwen3-coder-30b"
    assert {field.key: field.required for field in linkrra.credential_fields} == {
        "api_base": False,
        "api_key": True,
    }


@pytest.mark.asyncio
async def test_linkrra_is_listed_only_under_the_endpoints_it_serves():
    from litellm.proxy.public_endpoints.public_endpoints import get_supported_endpoints

    supported: Final = await get_supported_endpoints()
    listed_under: Final = {
        entry["key"]
        for entry in supported.model_dump()["endpoints"]
        if "linkrra" in {provider["slug"] for provider in entry["providers"]}
    }

    assert listed_under == {"chat_completions", "messages"}


def test_linkrra_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_API_BASE}/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="linkrra/qwen3-coder-30b",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="linkrra-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == f"{_API_BASE}/chat/completions"
    assert request.headers["authorization"] == "Bearer linkrra-test-key"
    assert body["model"] == "qwen3-coder-30b"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Linkrra"


def test_linkrra_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_API_BASE}/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="linkrra/qwen3-coder-30b",
            input="Say hello",
            api_key="linkrra-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == f"{_API_BASE}/chat/completions"
    assert request.headers["authorization"] == "Bearer linkrra-test-key"
    assert body["model"] == "qwen3-coder-30b"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.output[0].content[0].text == "Hello from Linkrra"


@pytest.mark.asyncio
async def test_linkrra_anthropic_messages_request_is_sent_natively(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{_API_BASE}/messages").respond(
            200,
            json={
                "id": "msg_linkrra",
                "type": "message",
                "role": "assistant",
                "model": "qwen3-coder-30b",
                "content": [{"type": "text", "text": "Hello from Linkrra"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model="linkrra/qwen3-coder-30b",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="linkrra-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == f"{_API_BASE}/messages"
    assert request.headers["authorization"] == "Bearer linkrra-test-key"
    assert body["model"] == "qwen3-coder-30b"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 32
    assert response["content"][0]["text"] == "Hello from Linkrra"
