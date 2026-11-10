from types import SimpleNamespace
from typing import Final

import httpx
import pytest
from pydantic import JsonValue, TypeAdapter
from respx import MockRouter

import litellm
from litellm.google_genai import agenerate_content_stream
from litellm.llms.gemini.google_genai.transformation import GoogleGenAIConfig
from litellm.llms.vertex_ai.common_utils import GEMINI_MAX_ENUM_CHARACTERS

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


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ([], r"argument after \*\* must be a mapping, not list"),
        (["candidates"], "list indices must be integers or slices, not str"),
    ],
)
def test_generate_content_response_non_object_body_raises_the_error_of_the_failing_lookup(
    body: JsonValue, message: str
) -> None:
    with pytest.raises(TypeError, match=message) as raised:
        GoogleGenAIConfig().transform_generate_content_response(
            model="gemini-2.5-flash",
            raw_response=httpx.Response(200, json=body),
            logging_obj=SimpleNamespace(model_call_details={}),
        )

    assert raised.type is TypeError


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


@pytest.mark.parametrize(
    ("model", "expected_level"),
    [
        ("gemini-flash-latest", {"type": "integer", "enum": ["1", "2", "5"]}),
        ("gemini-1.5-pro", {"type": "integer"}),
    ],
)
def test_normalize_response_schema_sends_integer_enums_to_gemini_3(
    model: str, expected_level: dict[str, object]
) -> None:
    config_dict: Final = {
        "responseSchema": {"type": "object", "properties": {"level": {"type": "integer", "enum": [1, 2, 5]}}}
    }

    GoogleGenAIConfig._normalize_response_schema(config_dict, model)

    assert config_dict["responseSchema"]["properties"]["level"] == expected_level


@pytest.mark.parametrize(
    ("value_count", "integer_enum_sent"),
    [(GEMINI_MAX_ENUM_CHARACTERS // 10, True), (GEMINI_MAX_ENUM_CHARACTERS // 10 + 1, False)],
)
def test_normalize_response_schema_drops_integer_enums_past_the_enum_budget(
    value_count: int, integer_enum_sent: bool
) -> None:
    values: Final = [10**9 + n for n in range(value_count)]
    config_dict: Final = {
        "responseSchema": {"type": "object", "properties": {"level": {"type": "integer", "enum": values}}}
    }

    GoogleGenAIConfig._normalize_response_schema(config_dict, "gemini-flash-latest")

    assert config_dict["responseSchema"]["properties"]["level"] == (
        {"type": "integer", "enum": [str(value) for value in values]}
        if integer_enum_sent
        else {"type": "integer"}
    )
