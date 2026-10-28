import json
from collections.abc import AsyncIterable
from typing import Final

import httpx
import pytest
import respx
from openai.types.responses.response_create_params import ResponseInputParam
from pydantic import ConfigDict, TypeAdapter

import litellm
from litellm.types.llms.openai import ResponseCompletedEvent, ResponsesAPIResponse
from tests.unit.proxy.conftest import httpx_transport

pytestmark: Final = pytest.mark.usefixtures(httpx_transport.__name__)
_GEMINI_URL: Final = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:(?:generateContent|streamGenerateContent).*"
)
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])


def _function_call_response(signature: str) -> dict[str, object]:
    return {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"location": "San Francisco"},
                            },
                            "thoughtSignature": signature,
                        }
                    ],
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 4,
            "candidatesTokenCount": 2,
            "totalTokenCount": 6,
        },
    }


@pytest.mark.asyncio
async def test_web_search_preview_is_sent_as_google_search() -> None:
    response_body: Final = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "Search completed"}]},
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 2,
            "candidatesTokenCount": 2,
            "totalTokenCount": 4,
        },
    }

    with respx.mock() as mock_router:
        route: Final = mock_router.post(url__regex=_GEMINI_URL).mock(
            return_value=httpx.Response(status_code=200, json=response_body)
        )
        response: Final = await litellm.aresponses(
            model="gemini/gemini-2.5-flash",
            api_key="test-key",
            input="Find current weather",
            tools=[{"type": "web_search_preview", "search_context_size": "low"}],
        )
        requests: Final = tuple(route.calls)

    assert isinstance(response, ResponsesAPIResponse)
    assert response.output_text == "Search completed"
    assert len(requests) == 1
    request_body: Final = _JSON_OBJECT.validate_json(requests[0].request.content)
    assert request_body["tools"] == [{"googleSearch": {}}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("assistant_content", "assistant_parts"),
    [
        ("Report read", [{"text": "Report read"}]),
        (
            [{"type": "input_audio", "input_audio": {"data": "c291bmQ=", "format": "wav"}}],
            [{"inline_data": {"data": "c291bmQ=", "mime_type": "audio/wav"}}],
        ),
    ],
)
async def test_media_history_survives_a_streaming_followup(
    assistant_content: str | list[dict[str, object]], assistant_parts: list[dict[str, object]]
) -> None:
    pdf_data: Final = "JVBERi0xLjQKJSB0b29sIHJlcG9ydAolJUVPRg=="
    initial_input: Final[ResponseInputParam] = [
        {"role": "user", "content": "Read the report"},
        {"type": "function_call", "call_id": "call_pdf", "name": "read_report", "arguments": "{}"},
        {
            "type": "function_call_output",
            "call_id": "call_pdf",
            "output": [
                {
                    "type": "input_file",
                    "filename": "report.pdf",
                    "file_data": f"data:application/pdf;base64,{pdf_data}",
                }
            ],
        },
    ]
    response_body: Final = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "Report read"}]},
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2, "totalTokenCount": 6},
    }
    with respx.mock() as mock_router:
        route: Final = mock_router.post(url__regex=_GEMINI_URL).mock(
            side_effect=[
                httpx.Response(200, json=response_body),
                httpx.Response(
                    200,
                    content=f"data: {json.dumps(response_body)}\n\n",
                    headers={"content-type": "text/event-stream"},
                ),
            ]
        )
        response: Final = await litellm.aresponses(
            model="gemini/gemini-2.5-flash",
            api_key="test-key",
            input=initial_input,
            tools=[{"type": "function", "name": "read_report", "parameters": {"type": "object"}}],
        )
        assert isinstance(response, ResponsesAPIResponse)
        assert response.output_text == "Report read"
        initial_request: Final = route.calls.last.request
        stream: Final = await litellm.aresponses(
            model="gemini/gemini-2.5-flash",
            api_key="test-key",
            stream=True,
            input=[
                *initial_input,
                {"role": "assistant", "content": assistant_content},
                {"role": "user", "content": "Read it again"},
            ],
            tools=[{"type": "function", "name": "read_report", "parameters": {"type": "object"}}],
        )
        stream_events: Final = TypeAdapter(
            AsyncIterable[object],
            config=ConfigDict(arbitrary_types_allowed=True),
        ).validate_python(stream)
        events: Final = tuple([event async for event in stream_events])
        contents: Final = tuple(
            TypeAdapter(list[dict[str, object]]).validate_python(
                _JSON_OBJECT.validate_json(request.content)["contents"]
            )
            for request in (initial_request, route.calls.last.request)
        )
    completed: Final = tuple(event for event in events if isinstance(event, ResponseCompletedEvent))
    assert tuple(event.response.output_text for event in completed) == ("Report read",)
    assert route.call_count == 2
    assert len(contents[0]) == 3
    assert contents[0][2] == {
        "role": "user",
        "parts": [
            {"function_response": {"name": "read_report", "response": {"content": ""}}},
            {"inline_data": {"data": pdf_data, "mime_type": "application/pdf"}},
        ],
    }
    assert contents[1][:3] == contents[0]
    assert contents[1][3:] == [
        {"role": "model", "parts": assistant_parts},
        {"role": "user", "parts": [{"text": "Read it again"}]},
    ]


@pytest.mark.asyncio
async def test_gemini_function_call_preserves_thought_signature() -> None:
    signature: Final = "gemini-thought-signature"
    response_body: Final = _function_call_response(signature)

    with respx.mock() as mock_router:
        route: Final = mock_router.post(url__regex=_GEMINI_URL).mock(
            return_value=httpx.Response(status_code=200, json=response_body)
        )
        response: Final = await litellm.aresponses(
            model="gemini/gemini-2.5-flash",
            api_key="test-key",
            input="What is the weather in San Francisco?",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                }
            ],
        )
        requests: Final = tuple(route.calls)

    assert isinstance(response, ResponsesAPIResponse)
    function_calls: Final = tuple(item for item in response.output if item.type == "function_call")
    assert len(function_calls) == 1
    assert function_calls[0].name == "get_weather"
    assert function_calls[0].provider_specific_fields == {"thought_signature": signature}
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_gemini_streaming_function_call_preserves_thought_signature() -> None:
    signature: Final = "gemini-stream-thought-signature"
    response_body: Final = _function_call_response(signature)
    event_stream: Final = f"data: {json.dumps(response_body)}\n\n"

    with respx.mock() as mock_router:
        route: Final = mock_router.post(url__regex=_GEMINI_URL).mock(
            return_value=httpx.Response(
                status_code=200,
                content=event_stream,
                headers={"content-type": "text/event-stream"},
            )
        )
        stream: Final = await litellm.aresponses(
            model="gemini/gemini-2.5-flash",
            api_key="test-key",
            input="What is the weather in San Francisco?",
            stream=True,
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                }
            ],
        )
        events: Final = tuple([event async for event in stream])
        requests: Final = tuple(route.calls)

    completed_events: Final = tuple(event for event in events if isinstance(event, ResponseCompletedEvent))
    assert len(completed_events) == 1
    function_calls: Final = tuple(item for item in completed_events[0].response.output if item.type == "function_call")
    assert len(function_calls) == 1
    assert function_calls[0].name == "get_weather"
    assert function_calls[0].provider_specific_fields == {"thought_signature": signature}
    assert len(requests) == 1
