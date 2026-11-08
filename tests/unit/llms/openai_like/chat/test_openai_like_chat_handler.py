import json
from datetime import datetime
from typing import Final

from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.exceptions import MidStreamFallbackError
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.openai_like.chat.handler import OpenAILikeChatHandler, make_call, make_sync_call
from litellm.llms.openai_like.common_utils import OpenAILikeError
from litellm.types.utils import ModelResponse

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


def _logging_obj(stream: bool) -> LiteLLMLogging:
    logging_obj: Final = LiteLLMLogging(
        model="mistral-medium-3",
        messages=[{"role": "user", "content": "hi"}],
        stream=stream,
        call_type="acompletion",
        start_time=datetime(2026, 1, 1),
        litellm_call_id="openai-like-timeout",
        function_id="openai-like-timeout",
    )
    logging_obj.update_environment_variables(
        model="mistral-medium-3", user=None, optional_params={}, litellm_params={}, custom_llm_provider="openai"
    )
    return logging_obj


def _complete(client: AsyncHTTPHandler | HTTPHandler, *, acompletion: bool, stream: bool, timeout: float | None):
    return OpenAILikeChatHandler().completion(
        model="mistral-medium-3",
        messages=[{"role": "user", "content": "hi"}],
        api_base="https://openai-like.example/v1",
        custom_llm_provider="openai",
        custom_prompt_dict={},
        model_response=ModelResponse(),
        print_verbose=print,
        encoding=None,
        api_key="sk-openai-like",
        logging_obj=_logging_obj(stream),
        optional_params={"stream": stream},
        acompletion=acompletion,
        timeout=timeout,
        client=client,
    )


@pytest.mark.asyncio
async def test_async_stream_sends_the_resolved_timeout_to_the_upstream_request() -> None:
    read_timeouts: Final[list[object]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        read_timeouts.append(request.extensions["timeout"]["read"])
        return _streamed_reply(request)

    stream: Final = await _complete(
        AsyncHTTPHandler(transport=httpx.MockTransport(respond)), acompletion=True, stream=True, timeout=8.0
    )
    chunks: Final = [chunk async for chunk in stream]

    assert read_timeouts == [8.0]
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == _LINE_SEPARATED_TEXT


class _StallsAfterFirstChunk(httpx.AsyncByteStream):
    async def __aiter__(self):
        chunk: Final = {"id": "chatcmpl-stall", "choices": [{"index": 0, "delta": {"content": "partial"}}]}
        yield f"data: {json.dumps(chunk)}\n\n".encode()
        raise httpx.ReadTimeout("Timeout on reading data from socket")


@pytest.mark.asyncio
async def test_async_stream_read_timeout_surfaces_as_a_408_timeout_the_router_can_fall_back_on() -> None:
    transport: Final = httpx.MockTransport(lambda request: httpx.Response(200, stream=_StallsAfterFirstChunk()))
    stream: Final = await _complete(AsyncHTTPHandler(transport=transport), acompletion=True, stream=True, timeout=8.0)

    with pytest.raises(MidStreamFallbackError) as raised:
        [chunk async for chunk in stream]

    assert raised.value.status_code == 408
    assert isinstance(raised.value.original_exception, litellm.Timeout)


def _read_timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("Read timed out", request=request)


@pytest.mark.asyncio
async def test_async_non_stream_read_timeout_maps_to_a_408() -> None:
    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(_read_timeout))

    with pytest.raises(OpenAILikeError) as raised:
        await _complete(client, acompletion=True, stream=False, timeout=8.0)

    assert raised.value.status_code == 408


def test_sync_non_stream_read_timeout_maps_to_a_408() -> None:
    client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_read_timeout)))

    with pytest.raises(OpenAILikeError) as raised:
        _complete(client, acompletion=False, stream=False, timeout=8.0)

    assert raised.value.status_code == 408
