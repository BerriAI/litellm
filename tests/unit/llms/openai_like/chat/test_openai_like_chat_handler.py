import json
from typing import Final
from unittest.mock import Mock

import httpx

from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.openai_like.chat.handler import make_sync_call
from litellm.utils import CustomStreamWrapper

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


def test_real_stream_keeps_reasoning_multi_tool_calls_and_terminal_usage():
    events = [
        "data: "
        + json.dumps(
            {
                "id": "c1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "glm-4",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "reasoning_content": "Thinking step 1"},
                        "finish_reason": None,
                    }
                ],
            }
        ),
        "data: "
        + json.dumps(
            {
                "id": "c2",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "glm-4",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                                },
                                {
                                    "index": 1,
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {"name": "get_time", "arguments": '{"city": "Paris"}'},
                                },
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        ),
        "data: "
        + json.dumps(
            {
                "id": "c3",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "glm-4",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "Here is the result"},
                        "finish_reason": None,
                    }
                ],
            }
        ),
        "data: "
        + json.dumps(
            {
                "id": "c4",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "glm-4",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        ),
        "data: "
        + json.dumps(
            {
                "id": "c5",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "glm-4",
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 64,
                    "total_tokens": 74,
                    "completion_tokens_details": {"reasoning_tokens": 30},
                },
            }
        ),
        "data: [DONE]",
    ]
    sse_bytes = ("\n\n".join(events) + "\n\n").encode("utf-8")

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse_bytes)

    mock_logging = Mock()
    mock_logging.model_call_details = {"litellm_params": {}}
    mock_logging.stream_options = {"include_usage": True}

    raw_stream = make_sync_call(
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond))),
        api_base="https://openai-like.example/v1/chat/completions",
        headers={},
        data=json.dumps({"model": "glm-4", "messages": []}),
        model="glm-4",
        messages=[],
        logging_obj=mock_logging,
        fake_stream=False,
    )
    wrapper = CustomStreamWrapper(
        completion_stream=raw_stream,
        model="glm-4",
        custom_llm_provider="vertex_ai",
        logging_obj=mock_logging,
        stream_options={"include_usage": True},
    )

    chunks = list(wrapper)
    assert len(chunks) == 5

    # 1. Reasoning-only delta must reach the client with reasoning_content and content=None
    reasoning_chunk = chunks[0]
    assert reasoning_chunk.choices[0].delta.reasoning_content == "Thinking step 1"
    assert reasoning_chunk.choices[0].delta.content is None

    # 2. Multi tool call delta must preserve both tool calls and their indices
    tool_chunk = chunks[1]
    assert len(tool_chunk.choices[0].delta.tool_calls) == 2
    assert tool_chunk.choices[0].delta.tool_calls[0].function.name == "get_weather"
    assert tool_chunk.choices[0].delta.tool_calls[0].index == 0
    assert tool_chunk.choices[0].delta.tool_calls[1].function.name == "get_time"
    assert tool_chunk.choices[0].delta.tool_calls[1].index == 1

    # 3. Content delta
    assert chunks[2].choices[0].delta.content == "Here is the result"

    # 4. Finish reason
    assert chunks[3].choices[0].finish_reason == "stop"

    # 5. Terminal usage chunk with provider usage details preserved
    assert chunks[4].usage is not None
    assert chunks[4].usage.completion_tokens == 64
    assert chunks[4].usage.prompt_tokens == 10
    assert chunks[4].usage.completion_tokens_details.reasoning_tokens == 30

