import json
from typing import Final

from unittest.mock import Mock

import httpx
import pytest

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.openai_like.chat.handler import make_call, make_sync_call

_FAKE_STREAMED_REPLY: Final = {
    "id": "chatcmpl-fake-stream",
    "object": "chat.completion",
    "created": 0,
    "model": "openai/gpt-oss-120b",
    "choices": [
        {
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "reasoning_content": "Look up both values for Paris.",
                "tool_calls": [
                    {
                        "id": "call_weather",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    },
                    {
                        "id": "call_time",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": '{"city": "Paris"}'},
                    },
                ],
            },
        }
    ],
    "usage": {
        "prompt_tokens": 85,
        "completion_tokens": 1955,
        "total_tokens": 2040,
        "prompt_tokens_details": {"cached_tokens": 64},
        "completion_tokens_details": {"reasoning_tokens": 1920},
    },
}


def test_fake_stream_keeps_reasoning_every_tool_call_and_token_detail():
    request_bodies: Final[list[dict[str, object]]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        request_bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_FAKE_STREAMED_REPLY)

    chunks: Final = list(
        make_sync_call(
            client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond))),
            api_base="https://openai-like.example/v1/chat/completions",
            headers={},
            data=json.dumps({"model": "openai/gpt-oss-120b", "messages": []}),
            model="openai/gpt-oss-120b",
            messages=[],
            logging_obj=Mock(),
            fake_stream=True,
        )
    )

    assert len(request_bodies) == 1
    assert len(chunks) == 1
    delta: Final = chunks[0].choices[0].delta
    assert delta.reasoning_content == "Look up both values for Paris."
    assert [call.function.name for call in delta.tool_calls] == ["get_weather", "get_time"]
    assert chunks[0].usage.prompt_tokens_details.cached_tokens == 64
    assert chunks[0].usage.completion_tokens_details.reasoning_tokens == 1920


_LINE_SEPARATED_TEXT: Final = "before\u2028between\u2029after"


def _streamed_reply(request: httpx.Request) -> httpx.Response:
    chunk: Final = {"id": "chatcmpl-separators", "choices": [{"index": 0, "delta": {"content": _LINE_SEPARATED_TEXT}}]}
    body: Final = f"data: {json.dumps(chunk, ensure_ascii=False)}\n\ndata: [DONE]\n\n"
    return httpx.Response(200, stream=httpx.ByteStream(body.encode()))


async def _streamed_chunks(sync_stream: bool) -> list[dict[str, object]]:
    call_args: Final = {
        "api_base": "https://openai-like.example/v1/chat/completions",
        "headers": {},
        "data": json.dumps({"model": "openai/gpt-oss-120b", "messages": [], "stream": True}),
        "model": "openai/gpt-oss-120b",
        "messages": [],
        "logging_obj": Mock(),
    }
    transport: Final = httpx.MockTransport(_streamed_reply)
    if sync_stream:
        return list(make_sync_call(client=HTTPHandler(client=httpx.Client(transport=transport)), **call_args))
    return [chunk async for chunk in await make_call(client=AsyncHTTPHandler(transport=transport), **call_args)]


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_stream", [True, False])
async def test_stream_keeps_unicode_line_separators_inside_the_delta(sync_stream: bool) -> None:
    chunks: Final = await _streamed_chunks(sync_stream)

    assert "".join(str(chunk["text"]) for chunk in chunks) == _LINE_SEPARATED_TEXT
