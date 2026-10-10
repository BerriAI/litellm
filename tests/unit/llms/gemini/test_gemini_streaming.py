import json
from collections.abc import Mapping
from itertools import chain
from typing import Final

import httpx
import pytest
import respx

import litellm


def _sse_body(chunks: tuple[Mapping[str, object], ...]) -> str:
    return "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"


@pytest.mark.asyncio
@respx.mock
async def test_async_gemini_streaming_tool_call_arguments_are_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    chunks: Final = (
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "generate_series_of_questions",
                                    "args": {
                                        "questions": [
                                            "What is a bridge?",
                                            "How is it designed?",
                                            "How is it built?",
                                        ]
                                    },
                                }
                            }
                        ],
                        "role": "model",
                    },
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 7,
                "totalTokenCount": 12,
            },
        },
        {
            "candidates": [
                {
                    "content": {"parts": [], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 7,
                "totalTokenCount": 12,
            },
        },
    )
    route: Final = respx.post(
        url__regex=r"https://generativelanguage\.googleapis\.com/v1beta/models/gemini-2\.5-flash-lite:streamGenerateContent.*"
    ).mock(
        return_value=httpx.Response(
            200,
            text=_sse_body(chunks),
            headers={"content-type": "text/event-stream"},
        )
    )

    stream: Final = await litellm.acompletion(
        model="gemini/gemini-2.5-flash-lite",
        messages=[
            {"role": "system", "content": "You are an AI assistant"},
            {"role": "user", "content": "Generate three questions about civil engineering."},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "generate_series_of_questions",
                    "description": "Generate questions",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "questions": {
                                "type": "array",
                                "items": {"type": "string"},
                            }
                        },
                        "required": ["questions"],
                    },
                },
            }
        ],
        stream=True,
        api_key="test-gemini-key",
    )
    response_chunks: Final = [chunk async for chunk in stream]
    tool_calls: Final = [
        tool_call
        for chunk in response_chunks
        for tool_call in chunk.choices[0].delta.tool_calls or ()
    ]

    assert route.call_count == 1
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "generate_series_of_questions"
    assert json.loads(tool_calls[0].function.arguments) == {
        "questions": [
            "What is a bridge?",
            "How is it designed?",
            "How is it built?",
        ]
    }


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
@respx.mock
async def test_gemini_streaming_legacy_function_call_arguments_are_valid(
    sync_mode: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    chunks: Final = (
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_current_weather",
                                    "args": {"location": "Boston", "unit": "fahrenheit"},
                                }
                            }
                        ],
                        "role": "model",
                    },
                    "index": 0,
                }
            ]
        },
        {
            "candidates": [
                {
                    "content": {"parts": [], "role": "model"},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ]
        },
    )
    respx.post(
        url__regex=r"https://generativelanguage\.googleapis\.com/v1beta/models/gemini-2\.5-flash-lite:streamGenerateContent.*"
    ).mock(
        return_value=httpx.Response(
            200,
            text=_sse_body(chunks),
            headers={"content-type": "text/event-stream"},
        )
    )
    messages: Final = [{"role": "user", "content": "What is the weather in Boston?"}]
    functions: Final = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string"},
                },
                "required": ["location", "unit"],
            },
        }
    ]
    if sync_mode:
        stream_chunks: Final = list(
            litellm.completion(
                model="gemini/gemini-2.5-flash-lite",
                messages=messages,
                functions=functions,
                stream=True,
                api_key="test-gemini-key",
            )
        )
    else:
        stream: Final = await litellm.acompletion(
            model="gemini/gemini-2.5-flash-lite",
            messages=messages,
            functions=functions,
            stream=True,
            api_key="test-gemini-key",
        )
        stream_chunks: Final = [chunk async for chunk in stream]

    function_calls: Final = tuple(
        chunk.choices[0].delta.function_call
        for chunk in stream_chunks
        if chunk.choices and chunk.choices[0].delta.function_call is not None
    )
    assert tuple(
        function_call.name for function_call in function_calls if function_call.name is not None
    ) == ("get_current_weather",)
    arguments: Final = "".join(function_call.arguments or "" for function_call in function_calls)
    assert json.loads(arguments) == {"location": "Boston", "unit": "fahrenheit"}
