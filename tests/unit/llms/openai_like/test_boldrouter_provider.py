import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache

BOLDROUTER_CHAT_URL: Final = "https://boldrouter.com/v1/chat/completions"


def _chat_completion(content: str) -> dict:
    return {
        "id": "chatcmpl-boldrouter",
        "object": "chat.completion",
        "created": 1_790_000_000,
        "model": "bold/auto",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 44,
            "total_tokens": 49,
            "completion_tokens_details": {"reasoning_tokens": 40},
        },
    }


def test_boldrouter_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("BOLDROUTER_API_KEY", "sk-bold-test")
    monkeypatch.delenv("BOLDROUTER_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="boldrouter/anthropic/claude-sonnet-4.6",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    # BoldRouter model IDs are themselves provider-prefixed; only the litellm prefix is stripped.
    assert model == "anthropic/claude-sonnet-4.6"
    assert provider == "boldrouter"
    assert api_key == "sk-bold-test"
    assert api_base == "https://boldrouter.com/v1"


def test_boldrouter_provider_reads_api_base_from_env(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("BOLDROUTER_API_KEY", "sk-bold-test")
    monkeypatch.setenv("BOLDROUTER_API_BASE", "https://gateway.internal.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="boldrouter/bold/auto",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "boldrouter"
    assert api_base == "https://gateway.internal.example/v1"


def test_boldrouter_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("BOLDROUTER_API_KEY", "sk-bold-env")

    _, provider, api_key, api_base = get_llm_provider(
        model="boldrouter/bold/auto",
        custom_llm_provider=None,
        api_base="https://proxy.internal.example/v1",
        api_key="sk-bold-explicit",
    )

    assert provider == "boldrouter"
    assert api_key == "sk-bold-explicit"
    assert api_base == "https://proxy.internal.example/v1"


def test_boldrouter_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    boldrouter = next(provider for provider in providers if provider["litellm_provider"] == "boldrouter")

    assert boldrouter["provider"] == "BOLDROUTER"
    assert boldrouter["provider_display_name"] == "BoldRouter"
    assert boldrouter["default_model_placeholder"] == "boldrouter/bold/auto"
    assert {field["key"]: field["required"] for field in boldrouter["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_boldrouter_supported_endpoints():
    package_root = Path(litellm.__file__).parent
    for matrix_path in (
        package_root / "provider_endpoints_support_backup.json",
        package_root.parent / "provider_endpoints_support.json",
    ):
        endpoints = json.loads(matrix_path.read_text())["providers"]["boldrouter"]["endpoints"]
        assert endpoints["chat_completions"] is True
        assert [name for name, supported in endpoints.items() if supported] == ["chat_completions"]


def test_boldrouter_chat_request_passes_routing_options_through(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(BOLDROUTER_CHAT_URL).respond(200, json=_chat_completion("Hello from BoldRouter"))
        response: Final = litellm.completion(
            model="boldrouter/bold/auto",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="sk-bold-test",
            extra_headers={"x-bold-routing": "price"},
            extra_body={"models": ["bold/auto", "openai/gpt-5.5"], "reasoning": {"effort": "high"}},
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == BOLDROUTER_CHAT_URL
    assert request.headers["authorization"] == "Bearer sk-bold-test"
    assert request.headers["x-bold-routing"] == "price"
    assert body["model"] == "bold/auto"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["models"] == ["bold/auto", "openai/gpt-5.5"]
    assert body["reasoning"] == {"effort": "high"}
    assert response.choices[0].message.content == "Hello from BoldRouter"
    assert response.usage.completion_tokens_details.reasoning_tokens == 40


@pytest.mark.asyncio
async def test_boldrouter_async_chat_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(BOLDROUTER_CHAT_URL).respond(200, json=_chat_completion("Hi"))
        response: Final = await litellm.acompletion(
            model="boldrouter/deepseek/deepseek-v4-flash",
            messages=[{"role": "user", "content": "Say hi"}],
            api_key="sk-bold-test",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert route.call_count == 1
    assert body["model"] == "deepseek/deepseek-v4-flash"
    assert response.choices[0].message.content == "Hi"
