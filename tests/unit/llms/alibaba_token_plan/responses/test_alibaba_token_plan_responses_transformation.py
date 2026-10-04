import json
from functools import partial
from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

MODEL: Final = "alibaba_token_plan/qwen3.8-max"
TOOLS: Final = (
    {"type": "web_search"},
    {"type": "web_extractor"},
    {"type": "code_interpreter"},
    {"type": "web_search_image"},
    {"type": "image_search"},
)
OUTPUT: Final = {
    "id": "response-token-plan",
    "created_at": 1,
    "object": "response",
    "model": "qwen3.8-max",
    "status": "completed",
    "output": [
        {"type": "web_extractor_call", "id": "tool-1", "status": "completed", "result": "Source material"},
        {
            "id": "msg-1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Source answer", "annotations": []}],
        },
    ],
    "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
}


class InjectedAsyncHTTPHandler(AsyncHTTPHandler):
    def __init__(self, client: httpx.AsyncClient) -> None:
        self.client = client


def responses_transport(request: httpx.Request, expected_url: str) -> httpx.Response:
    assert str(request.url) == expected_url
    assert request.headers["authorization"] == "Bearer token-plan-responses-test"
    payload: Final = json.loads(request.content)
    assert payload["model"] == "qwen3.8-max"
    assert payload["input"] == "Find sources"
    assert payload["tools"] == list(TOOLS)
    assert payload["previous_response_id"] == "previous-response"
    assert payload["enable_thinking"] is True
    if not payload.get("stream"):
        return httpx.Response(200, json=OUTPUT)
    events: Final = (
        {"type": "response.created", "response": {**OUTPUT, "status": "in_progress", "output": []}},
        {"type": "response.web_extractor_call.in_progress", "item_id": "tool-1", "output_index": 0},
        {
            "type": "response.output_text.delta",
            "item_id": "msg-1",
            "output_index": 1,
            "content_index": 0,
            "delta": "Source answer",
        },
        {"type": "response.completed", "response": OUTPUT},
    )
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        text="".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "api_base,expected_url",
    [
        (None, "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1/responses"),
        (
            "https://gateway.example/openai/v1",
            "https://gateway.example/openai/v1/responses",
        ),
    ],
)
async def test_native_responses_preserve_harness_tools_and_streams(
    monkeypatch: pytest.MonkeyPatch,
    is_async: bool,
    stream: bool,
    api_base: str | None,
    expected_url: str,
) -> None:
    monkeypatch.setenv("ALIBABA_TOKEN_PLAN_API_KEY", "token-plan-responses-test")
    transport: Final = httpx.MockTransport(partial(responses_transport, expected_url=expected_url))
    kwargs: Final = {
        "model": MODEL,
        "input": "Find sources",
        "tools": list(TOOLS),
        "previous_response_id": "previous-response",
        "stream": stream,
        "api_base": api_base,
        "extra_body": {"enable_thinking": True},
    }
    async with httpx.AsyncClient(transport=transport) as async_client:
        with httpx.Client(transport=transport) as sync_client:
            response: Final = (
                await litellm.aresponses(**kwargs, client=InjectedAsyncHTTPHandler(async_client))
                if is_async
                else litellm.responses(**kwargs, client=HTTPHandler(client=sync_client))
            )
            if stream:
                events: Final = (
                    [event.model_dump() async for event in response]
                    if is_async
                    else [event.model_dump() for event in response]
                )
                assert any(event["type"] == "response.web_extractor_call.in_progress" for event in events)
                assert "".join(event.get("delta", "") for event in events) == "Source answer"
                assert events[-1]["type"] == "response.completed"
            else:
                output: Final = response.model_dump()["output"]
                assert output[0]["type"] == "web_extractor_call"
                assert output[0]["result"] == "Source material"
                assert output[1]["content"][0]["text"] == "Source answer"
                assert response.usage.total_tokens == 5
