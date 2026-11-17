import json
from typing import Final

import httpx
import pytest
import respx
from pydantic import TypeAdapter

import litellm
from litellm.types.llms.openai import ResponseCompletedEvent, ResponsesAPIResponse
from tests.unit.proxy.conftest import httpx_transport

pytestmark: Final = pytest.mark.usefixtures(httpx_transport.__name__)
_GEMINI_URL: Final = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:(?:generateContent|streamGenerateContent).*"
)
_GEMINI_3_MODEL: Final = "gemini/gemini-3.8-flash"
_GEMINI_3_URL: Final = r"https://generativelanguage\.googleapis\.com/[^/]+/models/gemini-3\.8-flash:generateContent.*"
_GEMINI_3_REPLY: Final = {
    "candidates": [
        {"content": {"role": "model", "parts": [{"text": "How can I help?"}]}, "finishReason": "STOP", "index": 0}
    ],
    "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 4, "totalTokenCount": 8},
}
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


async def _gemini_contents_for(input_items: list[dict[str, object]], **kwargs: object) -> object:
    with respx.mock() as mock_router:
        route: Final = mock_router.post(url__regex=_GEMINI_3_URL).mock(
            return_value=httpx.Response(status_code=200, json=_GEMINI_3_REPLY)
        )
        response: Final = await litellm.aresponses(
            model=_GEMINI_3_MODEL, api_key="test-key", input=input_items, **kwargs
        )
        requests: Final = tuple(route.calls)

    assert isinstance(response, ResponsesAPIResponse)
    assert len(requests) == 1
    return _JSON_OBJECT.validate_json(requests[0].request.content)["contents"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "empty_user_item",
    [
        pytest.param({"role": "user", "content": ""}, id="empty_string"),
        pytest.param({"role": "user", "content": []}, id="empty_list"),
        pytest.param({"role": "user", "content": None}, id="null"),
        pytest.param({"role": "user"}, id="missing"),
    ],
)
async def test_an_empty_trailing_user_item_reaches_gemini_as_a_user_turn(empty_user_item: dict[str, object]) -> None:
    contents: Final = await _gemini_contents_for(
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}, empty_user_item]
    )

    assert contents == [
        {"role": "user", "parts": [{"text": "Hi"}]},
        {"role": "model", "parts": [{"text": "Hello!"}]},
        {"role": "user", "parts": [{"text": " "}]},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "empty_user_item",
    [
        pytest.param({"role": "user", "content": ""}, id="empty_string"),
        pytest.param({"role": "user", "content": []}, id="empty_list"),
        pytest.param({"role": "user", "content": None}, id="null"),
        pytest.param({"role": "user"}, id="missing"),
    ],
)
async def test_an_empty_trailing_user_item_reaches_an_alternating_gemini_deployment_as_the_configured_continue_message(
    empty_user_item: dict[str, object],
) -> None:
    contents: Final = await _gemini_contents_for(
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}, empty_user_item],
        ensure_alternating_roles=True,
        user_continue_message={"role": "user", "content": "Keep going"},
    )

    assert contents == [
        {"role": "user", "parts": [{"text": "Hi"}]},
        {"role": "model", "parts": [{"text": "Hello!"}]},
        {"role": "user", "parts": [{"text": "Keep going"}]},
    ]
