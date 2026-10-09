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


BOLDROUTER_MODELS: Final = tuple(sorted(name for name in litellm.model_cost if name.startswith("boldrouter/")))
BOLDROUTER_ROUTERS: Final = ("boldrouter/bold/auto", "boldrouter/bold/fast", "boldrouter/bold/frontier")


def test_boldrouter_backup_registry_mirrors_cost_map():
    package_root = Path(litellm.__file__).parent
    cost_map = json.loads((package_root.parent / "model_prices_and_context_window.json").read_text())
    backup = json.loads((package_root / "model_prices_and_context_window_backup.json").read_text())
    entries = {name: entry for name, entry in cost_map.items() if name.startswith("boldrouter/")}

    assert tuple(sorted(entries)) == BOLDROUTER_MODELS
    assert set(BOLDROUTER_ROUTERS) <= set(entries)
    assert entries == {name: backup[name] for name in entries}


@pytest.mark.parametrize("model", [m for m in BOLDROUTER_MODELS if m not in BOLDROUTER_ROUTERS])
def test_boldrouter_model_cost(model: str):
    from litellm.cost_calculator import cost_per_token

    prompt_cost, completion_cost = cost_per_token(
        model=model,
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
        custom_llm_provider="boldrouter",
    )
    model_info = litellm.get_model_info(model)

    assert model_info["litellm_provider"] == "boldrouter"
    assert model_info["mode"] == "chat"
    assert model_info["input_cost_per_token"] > 0
    assert prompt_cost == pytest.approx(model_info["input_cost_per_token"] * 1_000_000)
    assert completion_cost == pytest.approx(model_info["output_cost_per_token"] * 1_000_000)


@pytest.mark.parametrize("model", BOLDROUTER_ROUTERS)
def test_boldrouter_routers_support_tools_and_vision(model: str):
    assert litellm.supports_function_calling(model) is True
    assert litellm.supports_vision(model) is True


def test_boldrouter_chat_request_sends_tools(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    tool: Final = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
    tool_call_response: Final = _chat_completion("")
    tool_call_response["choices"][0]["message"] = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Zurich"}'},
            }
        ],
    }
    tool_call_response["choices"][0]["finish_reason"] = "tool_calls"
    with respx.mock() as upstream:
        route: Final = upstream.post(BOLDROUTER_CHAT_URL).respond(200, json=tool_call_response)
        response: Final = litellm.completion(
            model="boldrouter/bold/auto",
            messages=[{"role": "user", "content": "Weather in Zurich?"}],
            api_key="sk-bold-test",
            tools=[tool],
            tool_choice="auto",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert body["tools"] == [tool]
    assert body["tool_choice"] == "auto"
    assert response.choices[0].message.tool_calls[0].function.name == "get_weather"


def test_boldrouter_streaming_reports_usage(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    chunks: Final = [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {
            "choices": [],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 12,
                "total_tokens": 17,
                "completion_tokens_details": {"reasoning_tokens": 10},
            },
        },
    ]
    sse: Final = (
        "".join(
            "data: "
            + json.dumps(
                {
                    "id": "chatcmpl-boldrouter",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "bold/fast",
                    **c,
                }
            )
            + "\n\n"
            for c in chunks
        )
        + "data: [DONE]\n\n"
    )
    with respx.mock() as upstream:
        route: Final = upstream.post(BOLDROUTER_CHAT_URL).respond(
            200, content=sse.encode(), headers={"content-type": "text/event-stream"}
        )
        stream: Final = litellm.completion(
            model="boldrouter/bold/fast",
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="sk-bold-test",
            stream=True,
            stream_options={"include_usage": True},
        )
        text = ""
        usage = None
        for chunk in stream:
            if chunk.choices:
                text += chunk.choices[0].delta.content or ""
            usage = getattr(chunk, "usage", None) or usage

    body: Final = json.loads(route.calls.last.request.content)
    assert body["stream"] is True
    assert text == "Hello"
    assert usage is not None
    assert usage.completion_tokens == 12
    assert usage.completion_tokens_details.reasoning_tokens == 10


def test_boldrouter_transcription_is_rejected_before_any_request():
    with respx.mock() as upstream:
        with pytest.raises(Exception, match="Unmapped provider"):
            litellm.transcription(
                model="boldrouter/whisper-1",
                file=("audio.wav", b"RIFF0000WAVE", "audio/wav"),
                api_key="sk-bold-test",
            )
        assert upstream.calls.call_count == 0
