import asyncio
import base64
import copy
import json
import threading
from typing import Final

import httpx
import pytest

import litellm
from litellm import Router, acompletion
from litellm.litellm_core_utils.prompt_templates import image_handling
from litellm.llms.anthropic.pass_through.messages import handler as anthropic_messages_handler
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.gemini.common_utils import GoogleAIStudioTokenCounter
from litellm.llms.gemini.count_tokens.transformation import (
    GeminiCountTokensPayload,
    build_count_tokens_payload,
)
from litellm.llms.vertex_ai.gemini import transformation as gemini_transformation

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
_REMOTE_PNG_URL: Final = "https://img.example.com/cat.png"
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
            "messages": [
                _ASK,
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}}
                    ],
                },
            ]
        },
        id="tool-call-still-awaiting-its-result",
    ),
    pytest.param(
        "gemini-2.5-flash",
        {
            "tools": [
                {"type": "computer_20250124", "name": "computer", "display_width_px": 1024, "display_height_px": 768}
            ],
            "messages": [{"role": "user", "content": "open the settings page"}],
        },
        id="computer-use-tool",
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
    pytest.param(
        "gemini-2.5-flash",
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
                {"role": "user", "content": "c"},
            ]
        },
        id="assistant-text-blocks-without-other-anthropic-markers",
    ),
    pytest.param(
        "gemini-2.5-flash",
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [{"type": "text", "text": ""}]},
                {"role": "user", "content": "again"},
            ]
        },
        id="empty-assistant-text-block",
    ),
    pytest.param(
        "gemini-2.5-flash",
        {"system": "", "messages": [{"role": "user", "content": "hi"}]},
        id="empty-system-string",
    ),
    pytest.param(
        "gemini-2.5-flash",
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image", "source": {"type": "url", "url": _REMOTE_PNG_URL}},
                    ],
                }
            ]
        },
        id="https-image-url",
    ),
)


@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "request_body"), _MESSAGES_REQUESTS)
async def test_count_payload_is_what_v1_messages_sends_to_gemini(local_model_cost_map, model, request_body):
    image_handling.in_memory_cache.set_cache(_REMOTE_PNG_URL, f"data:image/png;base64,{_PNG}")
    sent: list[dict[str, object]] = []  # mutable-ok: the fake upstream appends each captured body
    await anthropic_messages_handler.anthropic_messages(
        max_tokens=16,
        model=f"gemini/{model}",
        custom_llm_provider="gemini",
        api_key="fake-gemini-key",
        client=_capturing_client(sent),
        **copy.deepcopy(request_body),
    )

    payload = await build_count_tokens_payload(
        model=model,
        messages=request_body["messages"],
        system=request_body.get("system"),
        tools=request_body.get("tools"),
        message_format="anthropic",
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
    pytest.param(
        {
            "messages": [
                {"role": "user", "content": "weather in Paris?"},
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
            "tools": [{"type": "web_search"}, _WEATHER_OPENAI],
        },
        id="openai-web-search-tool-with-tool-call-history",
    ),
    pytest.param(
        {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
                {"role": "user", "content": "c"},
            ]
        },
        id="assistant-text-parts",
    ),
    pytest.param(
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": _REMOTE_PNG_URL}},
                    ],
                }
            ]
        },
        id="https-image-url",
    ),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("request_body", _CHAT_REQUESTS)
async def test_count_payload_is_what_chat_completions_sends_to_gemini(local_model_cost_map, request_body):
    image_handling.in_memory_cache.set_cache(_REMOTE_PNG_URL, f"data:image/png;base64,{_PNG}")
    sent: list[dict[str, object]] = []  # mutable-ok: the fake upstream appends each captured body
    await acompletion(
        model="gemini/gemini-2.5-flash",
        api_key="fake-gemini-key",
        client=_capturing_client(sent),
        max_tokens=16,
        **copy.deepcopy(request_body),
    )

    payload = await build_count_tokens_payload(
        model="gemini-2.5-flash",
        messages=request_body["messages"],
        system=None,
        tools=request_body.get("tools"),
        message_format="openai",
    )

    assert isinstance(payload, GeminiCountTokensPayload), payload
    assert _as_wire(payload) == _counted_part_of(sent[-1])


_DEPLOYMENT_TOOL: Final = {
    "type": "function",
    "function": {
        "name": "lookup_ticket",
        "description": "Ticket lookup configured on the deployment",
        "parameters": {"type": "object", "properties": {"ticket_id": {"type": "string"}}},
    },
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("deployment_tools", "request_body"),
    (
        pytest.param([_DEPLOYMENT_TOOL], {"messages": [_ASK]}, id="deployment-tools-only"),
        pytest.param(
            [_DEPLOYMENT_TOOL],
            {"system": "be terse", "messages": [_ASK], "tools": [_WEATHER]},
            id="deployment-and-request-tools",
        ),
        pytest.param(
            [{**_WEATHER, "name": "lookup_ticket"}],
            {"messages": [_ASK], "tools": [_WEATHER, {"type": "web_search_20250305", "name": "web_search"}]},
            id="anthropic-deployment-tool-with-request-web-search",
        ),
    ),
)
async def test_count_includes_the_deployment_tools_the_router_sends(
    local_model_cost_map, deployment_tools, request_body
):
    deployment = {
        "model_name": "gemini-count",
        "litellm_params": {"model": "gemini/gemini-2.5-flash", "api_key": "fake-gemini-key", "tools": deployment_tools},
    }
    routed_request = copy.deepcopy(request_body)
    Router._merge_tools_from_deployment(deployment=deployment, kwargs=routed_request)
    sent: list[dict[str, object]] = []  # mutable-ok: the fake upstream appends each captured body
    await anthropic_messages_handler.anthropic_messages(
        max_tokens=16,
        model="gemini/gemini-2.5-flash",
        custom_llm_provider="gemini",
        api_key="fake-gemini-key",
        client=_capturing_client(sent),
        **routed_request,
    )

    counted: list[dict[str, object]] = []  # mutable-ok: the fake countTokens endpoint appends each body

    def count_tokens_endpoint(request: httpx.Request) -> httpx.Response:
        counted.append(json.loads(request.content)["generateContentRequest"])
        return httpx.Response(200, json={"totalTokens": 7})

    result = await GoogleAIStudioTokenCounter().count_anthropic_messages_tokens(
        model_to_use="gemini-2.5-flash",
        messages=copy.deepcopy(request_body["messages"]),
        contents=None,
        deployment=deployment,
        tools=copy.deepcopy(request_body.get("tools")),
        system=request_body.get("system"),
        client=httpx.AsyncClient(transport=httpx.MockTransport(count_tokens_endpoint)),
    )

    assert result is not None and result.error is not True, result
    count_body = counted[-1]
    assert {
        "contents": count_body["contents"],
        **({"system_instruction": count_body["systemInstruction"]} if "systemInstruction" in count_body else {}),
        **({"tools": count_body["tools"]} if "tools" in count_body else {}),
    } == _counted_part_of(sent[-1])


@pytest.mark.asyncio
async def test_build_count_tokens_payload_maps_openai_web_search_tool():
    payload = await build_count_tokens_payload(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": "hi"}],
        system=None,
        tools=[{"type": "web_search_preview"}],
        message_format="openai",
    )

    assert isinstance(payload, GeminiCountTokensPayload), payload
    assert payload.tools == ({"googleSearch": {}},)


@pytest.mark.asyncio
async def test_build_count_tokens_payload_wraps_responses_api_tool():
    payload = await build_count_tokens_payload(
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
        message_format="openai",
    )

    assert isinstance(payload, GeminiCountTokensPayload), payload
    assert payload.tools is not None
    function_declaration = payload.tools[0]["function_declarations"][0]
    assert function_declaration["name"] == "get_weather"
    assert function_declaration["parameters"] == {
        "type": "object",
        "properties": {"city": {"type": "string"}},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message_format", "image_block"),
    (
        pytest.param(
            "anthropic", {"type": "image", "source": {"type": "url", "url": "http://img.test/cat"}}, id="anthropic"
        ),
        pytest.param("openai", {"type": "image_url", "image_url": {"url": "http://img.test/cat"}}, id="openai"),
    ),
)
async def test_remote_image_is_fetched_without_blocking_the_event_loop(monkeypatch, message_format, image_block):
    ticks: list[float] = []  # mutable-ok: the ticker records each wake-up time
    ticks_while_fetching: list[int] = []  # mutable-ok: the image host records how far the ticker got during its sleep

    async def slow_image_host(request: httpx.Request) -> httpx.Response:
        before = len(ticks)
        await asyncio.sleep(0.3)
        ticks_while_fetching.append(len(ticks) - before)
        return httpx.Response(200, content=base64.b64decode(_PNG), headers={"content-type": "image/png"})

    image_client = AsyncHTTPHandler()
    image_client.client = httpx.AsyncClient(transport=httpx.MockTransport(slow_image_host))
    monkeypatch.setattr(litellm, "module_level_aclient", image_client)
    monkeypatch.setattr(litellm, "user_url_validation", False)
    loop = asyncio.get_running_loop()

    async def ticker() -> None:
        while True:
            ticks.append(loop.time())
            await asyncio.sleep(0.01)

    ticking = asyncio.create_task(ticker())
    try:
        payload = await build_count_tokens_payload(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": [{"type": "text", "text": "describe"}, image_block]}],
            system=None,
            tools=None,
            message_format=message_format,
        )
    finally:
        ticking.cancel()

    assert isinstance(payload, GeminiCountTokensPayload), payload
    assert payload.contents[0]["parts"][1]["inline_data"]["data"] == _PNG
    assert ticks_while_fetching and ticks_while_fetching[0] >= 10, ticks_while_fetching


_EXTENSIONLESS_GS_URI: Final = "gs://private-bucket/uploads/cat"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message_format", "image_block"),
    (
        pytest.param(
            "anthropic", {"type": "image", "source": {"type": "url", "url": _EXTENSIONLESS_GS_URI}}, id="anthropic"
        ),
        pytest.param("openai", {"type": "image_url", "image_url": {"url": _EXTENSIONLESS_GS_URI}}, id="openai"),
    ),
)
async def test_gcs_metadata_lookup_runs_off_the_event_loop(monkeypatch, message_format, image_block):
    lookup_threads: list[int] = []  # mutable-ok: the fake GCS lookup records the thread it ran on

    def lookup(image_url: str, vertex_project: str | None = None, vertex_credentials: object = None) -> str:
        lookup_threads.append(threading.get_ident())
        return "image/png"

    monkeypatch.setattr(gemini_transformation, "_get_gcs_object_content_type", lookup)

    payload = await build_count_tokens_payload(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": [{"type": "text", "text": "describe"}, image_block]}],
        system=None,
        tools=None,
        message_format=message_format,
    )

    assert payload.contents[0]["parts"][1]["file_data"]["file_uri"] == _EXTENSIONLESS_GS_URI
    assert lookup_threads and threading.get_ident() not in lookup_threads, lookup_threads
