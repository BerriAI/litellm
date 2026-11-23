import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_DEFAULT_API_BASE: Final = "https://api.kunavo.com/v1"
_CHAT_COMPLETIONS_URL: Final = f"{_DEFAULT_API_BASE}/chat/completions"
_MESSAGES_URL: Final = f"{_DEFAULT_API_BASE}/messages"
_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_kunavo",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "gpt-6-sol",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from Kunavo"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
}
_ANTHROPIC_MESSAGE: Final = {
    "id": "msg_kunavo",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-5-5",
    "content": [{"type": "text", "text": "Hello from Kunavo"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 4, "output_tokens": 3},
}
KUNAVO_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("kunavo/")))


def _lists_native_messages(model: str) -> bool:
    return "/v1/messages" in litellm.model_cost[model].get("supported_endpoints", ())


def test_kunavo_provider_resolution_reads_the_kunavo_key_and_default_base(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KUNAVO_API_KEY", "sk-kn-env")
    monkeypatch.delenv("KUNAVO_API_BASE", raising=False)

    resolved: Final = get_llm_provider(
        model="kunavo/claude-sonnet-5-5",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert resolved == ("claude-sonnet-5-5", "kunavo", "sk-kn-env", _DEFAULT_API_BASE)


def test_kunavo_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KUNAVO_API_KEY", "sk-kn-env")
    monkeypatch.setenv("KUNAVO_API_BASE", "https://kunavo.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model="kunavo/gpt-6-sol",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "kunavo"
    assert api_base == "https://kunavo.env.example/v1"


def test_kunavo_explicit_credentials_win_over_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("KUNAVO_API_KEY", "sk-kn-env")
    monkeypatch.setenv("KUNAVO_API_BASE", "https://kunavo.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="kunavo/gpt-6-sol",
        custom_llm_provider=None,
        api_base="https://kunavo.internal.example/v1",
        api_key="sk-kn-explicit",
    )

    assert provider == "kunavo"
    assert api_key == "sk-kn-explicit"
    assert api_base == "https://kunavo.internal.example/v1"


@pytest.mark.parametrize("model", ["claude-sonnet-5-5", "gpt-6-sol"])
def test_kunavo_api_base_keeps_the_provider_of_an_unprefixed_model(model: str):
    _, provider_without_api_base, _, _ = get_llm_provider(model=model)
    _, provider_with_kunavo_api_base, _, _ = get_llm_provider(
        model=model,
        api_base=_DEFAULT_API_BASE,
        api_key="sk-kn-test",
    )

    assert provider_with_kunavo_api_base == provider_without_api_base


def test_kunavo_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model="kunavo/gpt-6-sol",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=64,
            api_key="sk-kn-test",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer sk-kn-test"
    assert body["model"] == "gpt-6-sol"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["max_tokens"] == 64
    assert response.choices[0].message.content == "Hello from Kunavo"


def test_kunavo_chat_completion_streaming_request():
    chunks: Final = (
        'data: {"id":"chatcmpl_kunavo","object":"chat.completion.chunk","created":1790000000,'
        '"model":"claude-sonnet-5-5","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},'
        '"finish_reason":null}]}\n\n',
        'data: {"id":"chatcmpl_kunavo","object":"chat.completion.chunk","created":1790000000,'
        '"model":"claude-sonnet-5-5","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}}\n\n',
        "data: [DONE]\n\n",
    )
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(
            200,
            headers={"content-type": "text/event-stream"},
            content="".join(chunks).encode(),
        )
        response: Final = litellm.completion(
            model="kunavo/claude-sonnet-5-5",
            messages=[{"role": "user", "content": "Say hello"}],
            stream=True,
            api_key="sk-kn-test",
        )
        collected: Final = tuple(response)

    body: Final = json.loads(route.calls.last.request.content)
    assert route.call_count == 1
    assert body["stream"] is True
    assert body["model"] == "claude-sonnet-5-5"
    assert any(chunk.choices and chunk.choices[0].delta.content == "Hi" for chunk in collected)


def test_kunavo_chat_never_borrows_the_openai_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("KUNAVO_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-env")

    with respx.mock(assert_all_called=False) as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        with pytest.raises(litellm.AuthenticationError, match="KUNAVO_API_KEY"):
            litellm.completion(model="kunavo/gpt-6-sol", messages=[{"role": "user", "content": "Say hello"}])

    assert route.call_count == 0


@pytest.mark.asyncio
async def test_kunavo_claude_messages_reach_the_native_messages_endpoint(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(_MESSAGES_URL).respond(200, json=_ANTHROPIC_MESSAGE)
        response: Final = await litellm.anthropic.messages.acreate(
            model="kunavo/claude-sonnet-5-5",
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="sk-kn-test",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == _MESSAGES_URL
    assert request.headers["authorization"] == "Bearer sk-kn-test"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "claude-sonnet-5-5"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from Kunavo"


@pytest.mark.asyncio
@pytest.mark.parametrize("model", KUNAVO_MODELS)
async def test_kunavo_messages_route_matches_the_endpoints_of_the_cost_map_row(
    model: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock(assert_all_called=False) as upstream:
        native: Final = upstream.post(_MESSAGES_URL).respond(200, json=_ANTHROPIC_MESSAGE)
        chat: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = await litellm.anthropic.messages.acreate(
            model=model,
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="sk-kn-test",
        )

    expected_calls: Final = (1, 0) if _lists_native_messages(model) else (0, 1)
    assert (native.call_count, chat.call_count) == expected_calls, model
    assert response["content"][0]["text"] == "Hello from Kunavo"


@pytest.mark.asyncio
async def test_kunavo_native_messages_keep_the_one_hour_cache_ttl(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    one_hour_cache: Final = {"type": "ephemeral", "ttl": "1h"}

    with respx.mock() as upstream:
        route: Final = upstream.post(_MESSAGES_URL).respond(200, json=_ANTHROPIC_MESSAGE)
        await litellm.anthropic.messages.acreate(
            model="kunavo/claude-sonnet-5-5",
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "Say hello", "cache_control": one_hour_cache}]}
            ],
            max_tokens=32,
            api_key="sk-kn-test",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert body["messages"][0]["content"][0]["cache_control"] == one_hour_cache


def test_kunavo_responses_request_is_bridged_to_chat_completions():
    with respx.mock() as upstream:
        route: Final = upstream.post(_CHAT_COMPLETIONS_URL).respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.responses(
            model="kunavo/gpt-6-sol",
            input="Say hello",
            api_key="sk-kn-test",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert route.call_count == 1
    assert body["model"] == "gpt-6-sol"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    message_items: Final = [item for item in response.output if item.type == "message"]
    assert message_items[0].content[0].text == "Hello from Kunavo"


@pytest.mark.parametrize("model", KUNAVO_MODELS)
def test_kunavo_row_prices_the_call_and_answers_capability_lookups(model: str):
    from litellm.cost_calculator import cost_per_token

    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=1_000,
        completion_tokens=1_000,
        custom_llm_provider="kunavo",
    )
    model_info: Final = litellm.get_model_info(model)

    assert prompt_cost == pytest.approx(model_info["input_cost_per_token"] * 1_000)
    assert completion_cost == pytest.approx(model_info["output_cost_per_token"] * 1_000)
    assert model_info["litellm_provider"] == "kunavo"
    assert model_info["mode"] == "chat"
    assert model_info["max_tokens"] == model_info["max_output_tokens"] <= model_info["max_input_tokens"]
    assert 0 < model_info["cache_read_input_token_cost"] < model_info["input_cost_per_token"]
    capability_checks: Final = (
        ("supports_function_calling", litellm.supports_function_calling),
        ("supports_reasoning", litellm.supports_reasoning),
        ("supports_response_schema", litellm.supports_response_schema),
        ("supports_vision", litellm.supports_vision),
    )
    assert {name: check(model) for name, check in capability_checks} == {
        name: model_info.get(name) is True for name, _ in capability_checks
    }


@pytest.mark.usefixtures("local_model_cost_map")
def test_kunavo_wildcard_expands_to_every_kunavo_row():
    from litellm.proxy.auth.model_checks import get_known_models_from_wildcard

    assert KUNAVO_MODELS
    assert set(get_known_models_from_wildcard("kunavo/*")) == set(KUNAVO_MODELS)


def test_kunavo_backup_maps_mirror_the_root_maps():
    package_root: Final = Path(litellm.__file__).parent
    cost_map: Final = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    cost_backup: Final = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    endpoints: Final = json.loads((package_root.parent / "provider_endpoints_support.json").read_text())
    endpoints_backup: Final = json.loads((package_root / "provider_endpoints_support_backup.json").read_text())
    kunavo_rows: Final = {name: entry for name, entry in cost_map.items() if name.startswith("kunavo/")}

    assert tuple(sorted(kunavo_rows)) == KUNAVO_MODELS
    assert kunavo_rows == {name: cost_backup[name] for name in kunavo_rows}
    assert endpoints["providers"]["kunavo"] == endpoints_backup["providers"]["kunavo"]


def test_kunavo_is_available_in_add_model_form():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    kunavo: Final = next(provider for provider in providers if provider["litellm_provider"] == "kunavo")

    assert kunavo["provider"] == "KUNAVO"
    assert kunavo["default_model_placeholder"] in KUNAVO_MODELS
    assert {field["key"]: field["required"] for field in kunavo["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }
