import json
from pathlib import Path
from typing import Final

import pytest
import respx

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

STRAITLY_BASE: Final = "https://api.straitly.ai/v1"
MODEL: Final = "straitly/deepseek/deepseek-v4-1-flash-1017"
UPSTREAM_MODEL: Final = "deepseek/deepseek-v4-1-flash-1017"


def test_straitly_is_registered():
    assert litellm.LlmProviders.STRAITLY.value == "straitly"
    assert "straitly" in litellm.provider_list
    assert "straitly" in litellm.constants.openai_compatible_providers
    assert STRAITLY_BASE in litellm.constants.openai_compatible_endpoints


def test_straitly_provider_resolution_keeps_nested_model_id(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("STRAITLY_API_KEY", "straitly-test-key")
    monkeypatch.delenv("STRAITLY_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model=MODEL,
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == UPSTREAM_MODEL
    assert provider == "straitly"
    assert api_key == "straitly-test-key"
    assert api_base == STRAITLY_BASE


def test_straitly_api_base_env_overrides_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("STRAITLY_API_KEY", "straitly-test-key")
    monkeypatch.setenv("STRAITLY_API_BASE", "https://straitly.internal.example/v1")

    _, provider, _, api_base = get_llm_provider(model=MODEL, custom_llm_provider=None, api_base=None, api_key=None)

    assert provider == "straitly"
    assert api_base == "https://straitly.internal.example/v1"


def test_straitly_provider_keeps_explicit_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("STRAITLY_API_KEY", "straitly-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model=MODEL,
        custom_llm_provider=None,
        api_base="https://straitly.internal.example/v1",
        api_key="straitly-explicit-key",
    )

    assert provider == "straitly"
    assert api_key == "straitly-explicit-key"
    assert api_base == "https://straitly.internal.example/v1"


def test_straitly_add_model_form_placeholder_resolves_to_its_default_base():
    fields_path: Final = Path(litellm.__file__).parent / "proxy" / "public_endpoints" / "provider_create_fields.json"
    providers: Final = json.loads(fields_path.read_text())
    straitly: Final = next(provider for provider in providers if provider["litellm_provider"] == "straitly")
    fields_by_key: Final = {field["key"]: field for field in straitly["credential_fields"]}
    _, provider, _, api_base = get_llm_provider(model=straitly["default_model_placeholder"])

    assert straitly["provider_display_name"] == "Straitly"
    assert provider == "straitly"
    assert api_base == fields_by_key["api_base"]["placeholder"]
    assert fields_by_key["api_base"]["required"] is False
    assert fields_by_key["api_key"]["required"] is True
    assert fields_by_key["api_key"]["field_type"] == "password"


def test_straitly_chat_completion_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{STRAITLY_BASE}/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_straitly",
                "object": "chat.completion",
                "created": 1_795_640_000,
                "model": UPSTREAM_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from Straitly"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="straitly-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == f"{STRAITLY_BASE}/chat/completions"
    assert request.headers["authorization"] == "Bearer straitly-test-key"
    assert body["model"] == UPSTREAM_MODEL
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response.choices[0].message.content == "Hello from Straitly"


def test_straitly_forwards_tools_for_models_without_a_cost_map_row():
    tools: Final = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            },
        }
    ]
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{STRAITLY_BASE}/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_straitly_tools",
                "object": "chat.completion",
                "created": 1_795_640_000,
                "model": UPSTREAM_MODEL,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 9, "completion_tokens": 5, "total_tokens": 14},
            },
        )
        response: Final = litellm.completion(
            model=MODEL,
            messages=[{"role": "user", "content": "Weather in Paris?"}],
            tools=tools,
            tool_choice="auto",
            api_key="straitly-test-key",
        )

    body: Final = json.loads(route.calls.last.request.content)
    assert body["tools"] == tools
    assert body["tool_choice"] == "auto"
    assert response.choices[0].message.tool_calls[0].function.name == "get_weather"


def test_straitly_responses_request():
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{STRAITLY_BASE}/responses").respond(
            200,
            json={
                "id": "resp_straitly",
                "object": "response",
                "created_at": 1_795_640_000,
                "model": UPSTREAM_MODEL,
                "status": "completed",
                "output": [
                    {
                        "id": "msg_straitly",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "Hello from Straitly", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )
        response: Final = litellm.responses(
            model=MODEL,
            input="Say hello",
            api_key="straitly-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == f"{STRAITLY_BASE}/responses"
    assert request.headers["authorization"] == "Bearer straitly-test-key"
    assert body["model"] == UPSTREAM_MODEL
    assert body["input"] == "Say hello"
    assert response.output[0].content[0].text == "Hello from Straitly"


@pytest.mark.asyncio
async def test_straitly_anthropic_messages_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{STRAITLY_BASE}/messages").respond(
            200,
            json={
                "id": "msg_straitly",
                "type": "message",
                "role": "assistant",
                "model": UPSTREAM_MODEL,
                "content": [{"type": "text", "text": "Hello from Straitly"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model=MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="straitly-test-key",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert route.call_count == 1
    assert str(request.url) == f"{STRAITLY_BASE}/messages"
    assert request.headers["authorization"] == "Bearer straitly-test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert body["model"] == UPSTREAM_MODEL
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from Straitly"
