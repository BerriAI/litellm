import copy
import json
from typing import Final

import httpx
import pytest

from litellm import acompletion
from litellm.llms.anthropic.experimental_pass_through.messages import handler as anthropic_messages_handler
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.gemini.count_tokens.transformation import (
    GeminiCountTokensPayload,
    InvalidCountTokensRequest,
    build_count_tokens_payload,
    normalize_count_tokens_tools,
)

_GEMINI_REPLY: Final = {
    "candidates": [{"content": {"parts": [{"text": "ok"}], "role": "model"}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
}
_PNG: Final = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
_WEATHER: Final = {
    "name": "get_weather",
    "description": "Weather for a city",
    "input_schema": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "additionalProperties": False,
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}
_WEATHER_OPENAI: Final = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}
_ASK: Final = {"role": "user", "content": "Use get_weather for Paris."}
_TOOL_RESULT: Final = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "18C"}]}


def _capturing_client(sent: list[dict[str, object]]) -> AsyncHTTPHandler:
    def upstream(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(":generateContent"):
            sent.append(json.loads(request.content))
        return httpx.Response(200, json=_GEMINI_REPLY, request=request)

    client: Final = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    return client


def _counted_part_of(body: dict[str, object]) -> dict[str, object]:
    return {key: body[key] for key in ("contents", "system_instruction", "tools") if key in body}


def _as_wire(payload: GeminiCountTokensPayload) -> dict[str, object]:
    return json.loads(
        json.dumps(
            {
                "contents": list(payload.contents),
                **({"system_instruction": payload.system_instruction} if payload.system_instruction else {}),
                **({"tools": list(payload.tools)} if payload.tools else {}),
            }
        )
    )


_MESSAGES_REQUESTS: Final = (
    pytest.param("gemini-2.5-flash", {"messages": [{"role": "user", "content": "hello world"}]}, id="plain"),
    pytest.param(
        "gemini-2.5-flash",
        {"system": "You are terse.", "tools": [_WEATHER], "messages": [_ASK]},
        id="system-and-json-schema-tool",
    ),
    pytest.param(
        "gemini-2.5-flash",
        {
            "system": [{"type": "text", "text": "You are Claude Code."}, {"type": "text", "text": "Use tools."}],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": _PNG}},
                    ],
                }
            ],
        },
        id="system-blocks-and-image",
    ),
    pytest.param(
        "gemini-2.5-flash",
        {
            "tools": [_WEATHER],
            "messages": [
                _ASK,
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}
                    ],
                },
                _TOOL_RESULT,
            ],
        },
        id="tool-use-and-result",
    ),
    pytest.param(
        "gemini-2.5-flash",
        {
            "tools": [_WEATHER],
            "messages": [
                _ASK,
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "I should call get_weather for Paris.", "signature": ""},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "get_weather",
                            "input": {"city": "Paris"},
                            "provider_specific_fields": {"signature": "Cp4CAWkUfRNRLml7MiCU"},
                        },
                    ],
                },
                _TOOL_RESULT,
            ],
        },
        id="gemini-thinking-turn-as-claude-code-replays-it",
    ),
    pytest.param(
        "gemini-2.5-flash",
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "let me reason", "signature": "c2lnbmF0dXJl"},
                        {"type": "text", "text": "answer"},
                    ],
                },
                {"role": "user", "content": "go on"},
            ]
        },
        id="signed-thinking-block",
    ),
    pytest.param(
        "gemini-2.5-flash",
        {
            "messages": [
                {"role": "user", "content": "what is new in solar?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {"query": "solar"}},
                        {
                            "type": "web_search_tool_result",
                            "tool_use_id": "srv_1",
                            "content": [{"type": "web_search_result", "url": "https://e.org", "title": "t"}],
                        },
                        {"type": "text", "text": "Perovskite cells set a record."},
                    ],
                },
                {"role": "user", "content": "tell me more"},
            ]
        },
        id="litellm-synthesized-web-search-history",
    ),
    pytest.param(
        "gemini-2.5-flash",
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [{"type": "redacted_thinking", "data": "ZW5j"}, {"type": "text", "text": "hello"}],
                },
                {"role": "user", "content": "and now?"},
            ]
        },
        id="redacted-thinking",
    ),
    pytest.param(
        "gemini-2.5-flash",
        {
            "tools": [{"type": "code_execution_20250522", "name": "code_execution"}],
            "messages": [{"role": "user", "content": "compute 2**20"}],
        },
        id="code-execution-tool",
    ),
    pytest.param(
        "gemini-2.5-flash",
        {
            "tools": [{"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 1}],
            "messages": [{"role": "user", "content": "summarize e.org"}],
        },
        id="web-fetch-tool",
    ),
    pytest.param(
        "gemini-2.5-flash",
        {
            "tools": [{"type": "web_search_20250305", "name": "web_search", "max_uses": 1}],
            "messages": [{"role": "user", "content": "news"}],
        },
        id="web-search-tool",
    ),
    pytest.param(
        "gemini-2.5-flash",
        {
            "tools": [{"type": "web_search_20250305", "name": "web_search"}, _WEATHER],
            "messages": [_ASK],
        },
        id="web-search-mixed-with-function",
    ),
    pytest.param(
        "gemini-2.5-flash",
        {
            "tools": [{**_WEATHER, "name": "mcp__inventory_service__look_up_current_stock_level_for_a_warehouse_sku"}],
            "messages": [_ASK],
        },
        id="tool-name-over-64-chars",
    ),
    pytest.param(
        "gemini-2.5-flash",
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "JVBERi0xLjQKJcfsj6IK",
                            },
                        },
                        {"type": "text", "text": "what grew?"},
                    ],
                }
            ]
        },
        id="pdf-document",
    ),
    pytest.param(
        "gemini-2.5-flash",
        {
            "tools": [_WEATHER],
            "messages": [
                _ASK,
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ""},
                        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}},
                    ],
                },
                _TOOL_RESULT,
            ],
        },
        id="empty-text-block-beside-tool-use",
    ),
    pytest.param(
        "gemini-2.5-flash",
        {
            "tools": [_WEATHER],
            "messages": [
                _ASK,
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "functions.get_weather:0",
                            "name": "get_weather",
                            "input": {"city": "Paris"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "functions.get_weather:0", "content": "18C"}],
                },
            ],
        },
        id="cross-provider-tool-id",
    ),
    pytest.param(
        "gemini-1.5-flash",
        {"system": "be nice", "messages": [{"role": "user", "content": "hi"}]},
        id="model-without-system-support",
    ),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "request_body"), _MESSAGES_REQUESTS)
async def test_count_payload_is_what_v1_messages_sends_to_gemini(local_model_cost_map, model, request_body):
    sent: list[dict[str, object]] = []  # mutable-ok: the fake upstream appends each captured body
    await anthropic_messages_handler.anthropic_messages(
        max_tokens=16,
        model=f"gemini/{model}",
        custom_llm_provider="gemini",
        api_key="fake-gemini-key",
        client=_capturing_client(sent),
        **copy.deepcopy(request_body),
    )

    payload = build_count_tokens_payload(
        model=model,
        messages=request_body["messages"],
        system=request_body.get("system"),
        tools=request_body.get("tools"),
    )

    assert isinstance(payload, GeminiCountTokensPayload), payload
    assert _as_wire(payload) == _counted_part_of(sent[-1])


_CHAT_REQUESTS: Final = (
    pytest.param({"messages": [{"role": "user", "content": "hello world"}]}, id="plain"),
    pytest.param(
        {
            "messages": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "weather?"}],
            "tools": [_WEATHER_OPENAI],
        },
        id="system-and-function-tool",
    ),
    pytest.param(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "weather in Paris? and this image"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_PNG}"}},
                    ],
                },
                {
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
                {"role": "tool", "tool_call_id": "call_1", "content": "18C"},
            ],
            "tools": [_WEATHER_OPENAI],
        },
        id="tool-calls-results-and-data-url-image",
    ),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("request_body", _CHAT_REQUESTS)
async def test_count_payload_is_what_chat_completions_sends_to_gemini(local_model_cost_map, request_body):
    sent: list[dict[str, object]] = []  # mutable-ok: the fake upstream appends each captured body
    await acompletion(
        model="gemini/gemini-2.5-flash",
        api_key="fake-gemini-key",
        client=_capturing_client(sent),
        max_tokens=16,
        **copy.deepcopy(request_body),
    )

    payload = build_count_tokens_payload(
        model="gemini-2.5-flash",
        messages=request_body["messages"],
        system=None,
        tools=request_body.get("tools"),
    )

    assert isinstance(payload, GeminiCountTokensPayload), payload
    assert _as_wire(payload) == _counted_part_of(sent[-1])


def test_build_count_tokens_payload_maps_openai_web_search_tool():
    payload = build_count_tokens_payload(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "hi"}],
        system=None,
        tools=[{"type": "web_search_preview"}],
    )

    assert payload.tools == ({"googleSearch": {}},)


def test_build_count_tokens_payload_routes_openai_tool_types_to_openai_path():
    payload = build_count_tokens_payload(
        model="gemini-2.5-flash",
        messages=[
            {"role": "user", "content": "check it"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                    }
                ],
            },
            {"role": "tool", "content": "sunny", "tool_call_id": "call_1"},
        ],
        system=None,
        tools=[
            {"type": "web_search_preview"},
            {"type": "computer_use", "display_width": 1024, "display_height": 768},
        ],
    )

    function_call = payload.contents[1]["parts"][0].get("function_call")
    assert function_call == {"name": "get_weather", "args": {"city": "Paris"}}
    function_response = payload.contents[2]["parts"][0].get("function_response")
    assert function_response["name"] == "get_weather"


def test_build_count_tokens_payload_wraps_responses_api_tool():
    payload = build_count_tokens_payload(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "hi"}],
        system=None,
        tools=[
            {
                "type": "function",
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }
        ],
    )

    assert payload.tools is not None
    function_declaration = payload.tools[0]["function_declarations"][0]
    assert function_declaration["name"] == "get_weather"
    assert function_declaration["parameters"] == {
        "type": "object",
        "properties": {"city": {"type": "string"}},
    }


def test_build_count_tokens_payload_rejects_a_tool_result_without_its_tool_call():
    payload = build_count_tokens_payload(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": [{"type": "tool_result", "content": "18C"}]}],
        system=None,
        tools=None,
    )

    assert isinstance(payload, InvalidCountTokensRequest)
    assert "Missing corresponding tool call" in payload.message


def test_normalize_count_tokens_tools_handles_each_tool_shape():
    assert normalize_count_tokens_tools(model="gemini-2.5-flash", tools=None) is None
    assert normalize_count_tokens_tools(
        model="gemini-2.5-flash", tools=[{"function_declarations": [{"name": "g"}]}]
    ) == ({"function_declarations": [{"name": "g"}]},)
    assert normalize_count_tokens_tools(model="gemini-2.5-flash", tools=[{"googleSearch": {}}]) == (
        {"googleSearch": {}},
    )
    assert normalize_count_tokens_tools(
        model="gemini-2.5-flash",
        tools=[{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
    ) == ({"function_declarations": [{"name": "f", "parameters": {"type": "object"}}]},)
    assert normalize_count_tokens_tools(
        model="gemini-2.5-flash", tools=[{"name": "f", "input_schema": {"type": "object"}}]
    ) == ({"function_declarations": [{"name": "f", "parameters": {"type": "object"}}]},)
    assert normalize_count_tokens_tools(
        model="gemini-2.5-flash",
        tools=[{"googleSearch": {}}, {"type": "function", "function": {"name": "f"}}],
    ) == ({"function_declarations": [{"name": "f"}]},)
