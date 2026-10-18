from types import SimpleNamespace
from typing import Final

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter
from respx import MockRouter

import litellm
from litellm.google_genai import agenerate_content_stream
from litellm.llms.gemini.google_genai.transformation import GoogleGenAIConfig

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_PROMPT: Final = "Hello, can you tell me a short joke?"
_CONTENTS: Final = [{"role": "user", "parts": [{"text": _PROMPT}]}]
_FUNCTION_ARGS: Final = {
    "attendees": ["Bob", "Alice"],
    "date": "2025-03-27",
    "time": "10:00",
    "topic": "Q3 planning",
}


@pytest.mark.parametrize("model", ("gemini-2.5-flash", "vertex_ai/gemini-2.5-flash"))
def test_generate_content_request_maps_provider_parameters(model: str) -> None:
    request: Final = GoogleGenAIConfig().transform_generate_content_request(
        model=model,
        contents=_CONTENTS,
        tools=None,
        generate_content_config_dict={"temperature": 0.4, "topP": 0.8, "topK": 12},
    )

    assert _JSON_OBJECT.validate_python(request) == {
        "model": model,
        "contents": _CONTENTS,
        "tools": None,
        "generationConfig": {"temperature": 0.4, "topP": 0.8, "topK": 12},
    }


@pytest.mark.parametrize("model", ("gemini-2.5-flash", "vertex_ai/gemini-2.5-flash"))
def test_generate_content_response_keeps_text_usage_and_logging_response(model: str) -> None:
    raw_response: Final = httpx.Response(
        200,
        json={
            "responseId": "scripted-response",
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "Scientists trust atoms."}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 15, "candidatesTokenCount": 5, "totalTokenCount": 20},
        },
    )
    logging_obj: Final = SimpleNamespace(model_call_details={})
    response: Final = GoogleGenAIConfig().transform_generate_content_response(
        model=model,
        raw_response=raw_response,
        logging_obj=logging_obj,
    )

    assert _JSON_OBJECT.validate_python(response.candidates[0]) == {
        "content": {"role": "model", "parts": [{"text": "Scientists trust atoms."}]},
        "finishReason": "STOP",
    }
    assert _JSON_OBJECT.validate_python(response.model_dump(by_alias=True))["usageMetadata"] == {
        "promptTokenCount": 15,
        "candidatesTokenCount": 5,
        "totalTokenCount": 20,
    }
    assert logging_obj.model_call_details == {"httpx_response": raw_response}


def test_generate_content_response_parses_function_call() -> None:
    raw_response: Final = httpx.Response(
        200,
        json={
            "responseId": "scripted-tool-call",
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"functionCall": {"name": "schedule_meeting", "args": _FUNCTION_ARGS}}],
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 15, "candidatesTokenCount": 5, "totalTokenCount": 20},
        },
    )
    logging_obj: Final = SimpleNamespace(model_call_details={})
    response: Final = GoogleGenAIConfig().transform_generate_content_response(
        model="gemini-2.5-flash",
        raw_response=raw_response,
        logging_obj=logging_obj,
    )

    assert _JSON_OBJECT.validate_python(response.candidates[0]) == {
        "content": {
            "role": "model",
            "parts": [{"functionCall": {"name": "schedule_meeting", "args": _FUNCTION_ARGS}}],
        },
        "finishReason": "STOP",
    }


def test_generate_content_request_preserves_tool_declarations() -> None:
    tools: Final = [
        {
            "functionDeclarations": [
                {
                    "name": "schedule_meeting",
                    "description": "Schedules a meeting.",
                    "parameters": {
                        "type": "object",
                        "properties": {"topic": {"type": "string"}},
                        "required": ["topic"],
                    },
                }
            ]
        }
    ]
    request: Final = GoogleGenAIConfig().transform_generate_content_request(
        model="gemini-2.5-flash",
        contents=_CONTENTS,
        tools=tools,
        generate_content_config_dict={"temperature": 0.4},
    )

    assert _JSON_OBJECT.validate_python(request) == {
        "model": "gemini-2.5-flash",
        "contents": _CONTENTS,
        "tools": tools,
        "generationConfig": {"temperature": 0.4},
    }


@pytest.mark.asyncio
async def test_generate_content_stream_returns_text_and_usage(
    respx_mock: MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
    ).mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"responseId":"scripted-stream","candidates":[{"content":{"role":"model","parts":'
                b'[{"text":"Scientists trust "}]}}]}\n\n'
                b'data: {"responseId":"scripted-stream","candidates":[{"content":{"role":"model","parts":'
                b'[{"text":"atoms."}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":15,'
                b'"candidatesTokenCount":5,"totalTokenCount":20}}\n\n'
            ),
        )
    )

    chunks: Final = tuple(
        [
            chunk
            async for chunk in await agenerate_content_stream(
                model="gemini/gemini-2.5-flash",
                contents=_CONTENTS,
                api_key="gemini-offline",
            )
        ]
    )
    frames: Final = tuple(_JSON_OBJECT.validate_json(chunk.removeprefix(b"data: ").strip()) for chunk in chunks)

    assert frames == (
        {
            "responseId": "scripted-stream",
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": "Scientists trust "}]}}
            ],
        },
        {
            "responseId": "scripted-stream",
            "candidates": [{"content": {"role": "model", "parts": [{"text": "atoms."}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 15, "candidatesTokenCount": 5, "totalTokenCount": 20},
        },
    )
    assert _JSON_OBJECT.validate_json(route.calls.last.request.content) == {
        "model": "gemini-2.5-flash",
        "contents": _CONTENTS,
        "tools": None,
        "generationConfig": {},
    }


@pytest.mark.asyncio
async def test_generate_content_stream_parses_function_call(
    respx_mock: MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
    ).mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"responseId":"scripted-tool-call","candidates":[{"content":{"role":"model","parts":'
                b'[{"functionCall":{"name":"schedule_meeting","args":{"attendees":["Bob","Alice"],'
                b'"date":"2025-03-27","time":"10:00","topic":"Q3 planning"}}}]}}],"usageMetadata":'
                b'{"promptTokenCount":15,"candidatesTokenCount":5,"totalTokenCount":20}}\n\n'
            ),
        )
    )
    tools: Final = [{"functionDeclarations": [{"name": "schedule_meeting", "parameters": {"type": "object"}}]}]

    chunks: Final = tuple(
        [
            chunk
            async for chunk in await agenerate_content_stream(
                model="gemini/gemini-2.5-flash",
                contents=_CONTENTS,
                tools=tools,
                api_key="gemini-offline",
            )
        ]
    )
    frames: Final = tuple(_JSON_OBJECT.validate_json(chunk.removeprefix(b"data: ").strip()) for chunk in chunks)

    assert frames == (
        {
            "responseId": "scripted-tool-call",
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"functionCall": {"name": "schedule_meeting", "args": _FUNCTION_ARGS}}],
                    }
                }
            ],
            "usageMetadata": {"promptTokenCount": 15, "candidatesTokenCount": 5, "totalTokenCount": 20},
        },
    )
    assert _JSON_OBJECT.validate_json(route.calls.last.request.content)["tools"] == tools
