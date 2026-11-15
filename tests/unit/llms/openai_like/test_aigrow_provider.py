import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "https://aigrow.gg/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_MESSAGES_URL: Final = f"{_DEFAULT_API_BASE}/messages"
_NATIVE_MESSAGES_MODELS: Final = ("claude-fable-5.2", "claude-fable-5.1", "claude-opus-5.5", "claude-sonnet-5.5")
_BRIDGED_MESSAGES_MODELS: Final = ("gpt-6-astra", "gpt-6.1-sol", "gpt-6.1-terra", "gemini-4-argon", "grok-4.7")
_UNMAPPED_MODEL: Final = "a-model-not-in-the-cost-map"
_WEATHER_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_aigrow",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "gpt-6.1-sol",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from AIGROW"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}
_ANTHROPIC_MESSAGE: Final = {
    "id": "msg_aigrow",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5.5",
    "content": [{"type": "text", "text": "Hello from AIGROW"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 4, "output_tokens": 3},
}


def test_aigrow_provider_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIGROW_API_KEY", "aigrow-test-key")
    monkeypatch.delenv("AIGROW_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="aigrow/claude-sonnet-5.5",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert (model, provider, api_key, api_base) == ("claude-sonnet-5.5", "aigrow", "aigrow-test-key", _DEFAULT_API_BASE)


def test_aigrow_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIGROW_API_KEY", "aigrow-env-key")
    monkeypatch.setenv("AIGROW_API_BASE", "https://aigrow.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="aigrow/claude-sonnet-5.5",
        custom_llm_provider=None,
        api_base="https://aigrow.internal.example/v1",
        api_key="aigrow-explicit-key",
    )

    assert (provider, api_key, api_base) == ("aigrow", "aigrow-explicit-key", "https://aigrow.internal.example/v1")


def test_aigrow_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIGROW_API_KEY", "aigrow-env-key")
    monkeypatch.setenv("AIGROW_API_BASE", "https://aigrow.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="aigrow/claude-sonnet-5.5",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert (provider, api_base) == ("aigrow", "https://aigrow.env.example/v1")


def test_aigrow_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIGROW_API_KEY", "aigrow-env-key")

    resolved: Final = get_llm_provider(
        model="gpt-6.1-sol",
        custom_llm_provider=None,
        api_base=_DEFAULT_API_BASE,
        api_key=None,
    )

    assert resolved == ("gpt-6.1-sol", "aigrow", "aigrow-env-key", _DEFAULT_API_BASE)


def test_aigrow_is_available_in_add_model_form() -> None:
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    aigrow: Final = next(provider for provider in providers if provider["litellm_provider"] == "aigrow")

    assert (aigrow["provider"], aigrow["provider_display_name"]) == ("AIGROW", "AIGROW")
    assert aigrow["default_model_placeholder"] in litellm.models_by_provider["aigrow"]
    assert {field["key"]: field["required"] for field in aigrow["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


# USD per 1M tokens as published at https://aigrow.gg/ai-api and in models.dev providers/aigrow, read 2026-11-14
@pytest.mark.parametrize(
    ("model", "input_rate", "output_rate", "cache_read_rate", "cache_write_rate"),
    [("claude-sonnet-5.5", 0.8, 4.0, 0.075, 1.6), ("gpt-6.1-sol", 0.5, 2.5, 0.1, 0.625)],
)
@pytest.mark.usefixtures("local_model_cost_map")
def test_aigrow_cost_per_token_bills_the_published_rates(
    model: str, input_rate: float, output_rate: float, cache_read_rate: float, cache_write_rate: float
) -> None:
    from litellm.cost_calculator import cost_per_token

    uncached: Final = cost_per_token(
        model=f"aigrow/{model}", prompt_tokens=1_000_000, completion_tokens=1_000_000, custom_llm_provider="aigrow"
    )
    cache_read: Final = cost_per_token(
        model=f"aigrow/{model}",
        prompt_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
        custom_llm_provider="aigrow",
    )
    cache_write: Final = cost_per_token(
        model=f"aigrow/{model}",
        prompt_tokens=1_000_000,
        cache_creation_input_tokens=1_000_000,
        custom_llm_provider="aigrow",
    )

    assert uncached == pytest.approx((input_rate, output_rate))
    assert cache_read == pytest.approx((cache_read_rate, 0.0))
    assert cache_write == pytest.approx((cache_write_rate, 0.0))


@pytest.mark.usefixtures("local_model_cost_map")
def test_aigrow_wildcard_expands_to_every_aigrow_row() -> None:
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    rows: Final = {name for name, entry in litellm.model_cost.items() if entry.get("litellm_provider") == "aigrow"}

    assert rows
    assert set(get_known_models_from_wildcard("aigrow/*")) == rows


@pytest.mark.parametrize("model", (*_NATIVE_MESSAGES_MODELS, *_BRIDGED_MESSAGES_MODELS, _UNMAPPED_MODEL))
def test_aigrow_chat_completion_forwards_tools_and_reasoning_effort(model: str) -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model=f"aigrow/{model}",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[_WEATHER_TOOL],
            tool_choice="auto",
            reasoning_effort="high",
            max_completion_tokens=64,
            api_key="aigrow-test-key",
        )

    request: Final = route.calls.last.request
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer aigrow-test-key"
    assert json.loads(request.content) == {
        "model": model,
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": [_WEATHER_TOOL],
        "tool_choice": "auto",
        "reasoning_effort": "high",
        "max_completion_tokens": 64,
    }
    assert response.choices[0].message.content == "Hello from AIGROW"


def test_aigrow_without_its_own_key_never_sends_the_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AIGROW_API_KEY", raising=False)
    monkeypatch.delenv("AIGROW_API_BASE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-only-key")
    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        with pytest.raises(litellm.AuthenticationError, match="AIGROW_API_KEY"):
            _ = litellm.completion(model="aigrow/gpt-6.1-sol", messages=[{"role": "user", "content": "Say hello"}])

    assert route.call_count == 0


def test_aigrow_responses_request_is_bridged_to_chat_completions() -> None:
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="aigrow/gpt-6.1-sol",
            input="Say hello",
            api_key="aigrow-test-key",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert route.call_count == 1
    assert (body["model"], body["messages"]) == ("gpt-6.1-sol", [{"role": "user", "content": "Say hello"}])
    message_items: Final = [item for item in response.output if item.type == "message"]
    assert message_items[0].content[0].text == "Hello from AIGROW"


@pytest.mark.asyncio
@pytest.mark.parametrize("model", _NATIVE_MESSAGES_MODELS)
async def test_aigrow_anthropic_messages_reach_native_messages_endpoint_for_claude(
    model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AIGROW_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock(assert_all_called=False) as upstream:
        native: Final = upstream.post(_MESSAGES_URL).respond(200, json=_ANTHROPIC_MESSAGE)
        chat: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model=f"aigrow/{model}",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="aigrow-test-key",
        )

    request: Final = native.calls.last.request
    body: Final = json.loads(request.content)
    assert (native.call_count, chat.call_count) == (1, 0)
    assert request.headers["authorization"] == "Bearer aigrow-test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert (body["model"], body["messages"], body["max_tokens"]) == (
        model,
        [{"role": "user", "content": "Say hello"}],
        32,
    )
    assert response["content"][0]["text"] == "Hello from AIGROW"


@pytest.mark.asyncio
@pytest.mark.parametrize("model", _BRIDGED_MESSAGES_MODELS)
async def test_aigrow_anthropic_messages_bridge_to_chat_completions_for_other_models(
    model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AIGROW_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock(assert_all_called=False) as upstream:
        native: Final = upstream.post(_MESSAGES_URL).respond(200, json=_ANTHROPIC_MESSAGE)
        chat: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model=f"aigrow/{model}",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="aigrow-test-key",
        )

    body: Final = json.loads(chat.calls.last.request.content)
    assert (native.call_count, chat.call_count) == (0, 1)
    assert chat.calls.last.request.headers["authorization"] == "Bearer aigrow-test-key"
    assert (body["model"], body["messages"], body["max_tokens"]) == (
        model,
        [{"role": "user", "content": "Say hello"}],
        32,
    )
    assert any(block.get("text") == "Hello from AIGROW" for block in response["content"])
