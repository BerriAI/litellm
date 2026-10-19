import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.cost_calculator import cost_per_token
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

_MODEL: Final = "unrestricted/qwen3.8-27b-uncensored"
_PROMPT_TOKENS: Final = 1_000
_CACHED_PROMPT_TOKENS: Final = 600
_COMPLETION_TOKENS: Final = 100

_CHAT_COMPLETION: Final = {
    "id": "chatcmpl_unrestricted",
    "object": "chat.completion",
    "created": 1_790_000_000,
    "model": "qwen3.8-27b-uncensored",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello from Unrestricted",
                "reasoning_content": "The user wants a greeting",
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": _PROMPT_TOKENS,
        "completion_tokens": _COMPLETION_TOKENS,
        "total_tokens": _PROMPT_TOKENS + _COMPLETION_TOKENS,
        "prompt_tokens_details": {"cached_tokens": _CACHED_PROMPT_TOKENS},
    },
}

UNRESTRICTED_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("unrestricted/")))


def test_unrestricted_is_a_registered_provider():
    assert litellm.LlmProviders.UNRESTRICTED.value == "unrestricted"
    assert "unrestricted" in litellm.provider_list
    assert "unrestricted" in litellm.constants.openai_compatible_providers


def test_unrestricted_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("UNRESTRICTED_API_KEY", "unrestricted-test-key")

    model, provider, api_key, api_base = get_llm_provider(
        model=_MODEL,
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "qwen3.8-27b-uncensored"
    assert provider == "unrestricted"
    assert api_key == "unrestricted-test-key"
    assert api_base == "https://api.unrestricted.sh/v1"


def test_unrestricted_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("UNRESTRICTED_API_KEY", "unrestricted-env-key")
    monkeypatch.setenv("UNRESTRICTED_API_BASE", "https://unrestricted.env.example/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model=_MODEL,
        custom_llm_provider=None,
        api_base="https://unrestricted.internal.example/v1",
        api_key="unrestricted-explicit-key",
    )

    assert provider == "unrestricted"
    assert api_key == "unrestricted-explicit-key"
    assert api_base == "https://unrestricted.internal.example/v1"


def test_unrestricted_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("UNRESTRICTED_API_KEY", "unrestricted-env-key")
    monkeypatch.setenv("UNRESTRICTED_API_BASE", "https://unrestricted.env.example/v1")

    _, provider, _, api_base = get_llm_provider(
        model=_MODEL,
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "unrestricted"
    assert api_base == "https://unrestricted.env.example/v1"


def test_unrestricted_api_base_autodetects_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("UNRESTRICTED_API_KEY", "unrestricted-env-key")

    model, provider, api_key, api_base = get_llm_provider(
        model="qwen3.8-27b-uncensored",
        custom_llm_provider=None,
        api_base="https://api.unrestricted.sh/v1",
        api_key=None,
    )

    assert model == "qwen3.8-27b-uncensored"
    assert provider == "unrestricted"
    assert api_key == "unrestricted-env-key"
    assert api_base == "https://api.unrestricted.sh/v1"


@pytest.mark.parametrize("model", UNRESTRICTED_MODELS)
def test_unrestricted_model_cost_and_capabilities(model: str):
    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        custom_llm_provider="unrestricted",
    )
    model_info = litellm.get_model_info(model)

    assert prompt_cost == pytest.approx(model_info["input_cost_per_token"] * 1_000_000)
    assert completion_cost == pytest.approx(model_info["output_cost_per_token"] * 1_000_000)
    assert 0 < model_info["cache_read_input_token_cost"] < model_info["input_cost_per_token"]
    assert model_info["output_cost_per_token"] > 0
    assert model_info["max_tokens"] == model_info["max_output_tokens"] <= model_info["max_input_tokens"]
    assert model_info["litellm_provider"] == "unrestricted"
    assert model_info["mode"] == "chat"
    assert model_info["supports_function_calling"] is True
    assert model_info["supports_native_streaming"] is True
    assert model_info["supports_reasoning"] is True
    assert model_info["supports_response_schema"] is True
    assert litellm.supports_vision(model) is model_info["supports_vision"]


def test_unrestricted_backup_registry_mirrors_cost_map():
    package_root = Path(litellm.__file__).parent
    cost_map = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    unrestricted_entries = {name: entry for name, entry in cost_map.items() if name.startswith("unrestricted/")}

    assert tuple(sorted(unrestricted_entries)) == UNRESTRICTED_MODELS
    assert unrestricted_entries
    assert unrestricted_entries == {name: backup[name] for name in unrestricted_entries}


def test_unrestricted_is_available_in_add_model_form():
    fields_path = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers = json.loads(fields_path.read_text())
    unrestricted = next(provider for provider in providers if provider["litellm_provider"] == "unrestricted")

    assert unrestricted["provider"] == "UNRESTRICTED"
    assert unrestricted["provider_display_name"] == "Unrestricted"
    assert unrestricted["default_model_placeholder"] in UNRESTRICTED_MODELS
    assert {field["key"]: field["required"] for field in unrestricted["credential_fields"]} == {
        "api_base": False,
        "api_key": True,
    }


def test_unrestricted_supported_endpoints():
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

    assert json.loads(backup_path.read_text())["providers"]["unrestricted"]["endpoints"] == expected
    assert json.loads(root_path.read_text())["providers"]["unrestricted"]["endpoints"] == expected


def test_unrestricted_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.unrestricted.sh/v1/chat/completions").respond(
            200, json=_CHAT_COMPLETION
        )
        response: Final = litellm.completion(
            model=_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="unrestricted-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.unrestricted.sh/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer unrestricted-test-key"
    assert body["model"] == "qwen3.8-27b-uncensored"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Unrestricted"
    assert response.choices[0].message.reasoning_content == "The user wants a greeting"


@pytest.mark.parametrize("reasoning_effort", litellm.model_cost[_MODEL]["reasoning_effort_levels"])
def test_unrestricted_forwards_each_declared_reasoning_effort(reasoning_effort: str):
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.unrestricted.sh/v1/chat/completions").respond(
            200, json=_CHAT_COMPLETION
        )
        litellm.completion(
            model=_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            reasoning_effort=reasoning_effort,
            api_key="unrestricted-test-key",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert body["reasoning_effort"] == reasoning_effort


def test_unrestricted_forwards_tools_and_json_schema_response_format():
    tools: Final = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            },
        }
    ]
    response_format: Final = {
        "type": "json_schema",
        "json_schema": {
            "name": "weather",
            "schema": {"type": "object", "properties": {"forecast": {"type": "string"}}, "required": ["forecast"]},
            "strict": True,
        },
    }
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.unrestricted.sh/v1/chat/completions").respond(
            200, json=_CHAT_COMPLETION
        )
        litellm.completion(
            model=_MODEL,
            messages=[{"role": "user", "content": "What is the weather in Oslo?"}],
            tools=tools,
            response_format=response_format,
            api_key="unrestricted-test-key",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert body["tools"] == tools
    assert body["response_format"] == response_format


def test_unrestricted_completion_cost_bills_cached_prompt_tokens_at_the_cache_read_rate():
    with respx.mock() as upstream:
        upstream.post("https://api.unrestricted.sh/v1/chat/completions").respond(200, json=_CHAT_COMPLETION)
        response: Final = litellm.completion(
            model=_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="unrestricted-test-key",
        )

    model_info: Final = litellm.get_model_info(_MODEL)
    expected_cost: Final = (
        (_PROMPT_TOKENS - _CACHED_PROMPT_TOKENS) * model_info["input_cost_per_token"]
        + _CACHED_PROMPT_TOKENS * model_info["cache_read_input_token_cost"]
        + _COMPLETION_TOKENS * model_info["output_cost_per_token"]
    )
    assert litellm.completion_cost(completion_response=response) == pytest.approx(expected_cost)


def test_unrestricted_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.unrestricted.sh/v1/responses").respond(
            200,
            json={
                "id": "resp_unrestricted",
                "object": "response",
                "created_at": 1_790_000_000,
                "model": "qwen3.8-27b-uncensored",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_unrestricted",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from Unrestricted", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model=_MODEL,
            input="Say hello",
            reasoning={"effort": "low"},
            api_key="unrestricted-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.unrestricted.sh/v1/responses"
    assert request.headers["authorization"] == "Bearer unrestricted-test-key"
    assert body["model"] == "qwen3.8-27b-uncensored"
    assert body["input"] == "Say hello"
    assert body["reasoning"] == {"effort": "low"}
    assert response.output[0].content[0].text == "Hello from Unrestricted"


@pytest.mark.asyncio
async def test_unrestricted_anthropic_messages_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    thinking: Final = {"type": "enabled", "budget_tokens": 1_024}
    with respx.mock() as upstream:
        route: Final = upstream.post("https://api.unrestricted.sh/v1/messages").respond(
            200,
            json={
                "id": "msg_unrestricted",
                "type": "message",
                "role": "assistant",
                "model": "qwen3.8-27b-uncensored",
                "content": [{"type": "text", "text": "Hello from Unrestricted"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model=_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=2_048,
            thinking=thinking,
            api_key="unrestricted-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == "https://api.unrestricted.sh/v1/messages"
    assert request.headers["authorization"] == "Bearer unrestricted-test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == "qwen3.8-27b-uncensored"
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert body["thinking"] == thinking
    assert response["content"][0]["text"] == "Hello from Unrestricted"
