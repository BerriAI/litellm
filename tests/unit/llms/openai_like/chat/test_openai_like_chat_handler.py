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


def test_real_stream_incremental_alternating_tool_calls_and_negative_controls():
    events: Final[list[str]] = [
        # Frame 1: index 0 initial (id, name, arg prefix)
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
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_weather",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": '{"city": "'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        ),
        # Frame 2: index 1 initial (id, name, arg prefix)
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
                                    "index": 1,
                                    "id": "call_time",
                                    "type": "function",
                                    "function": {"name": "get_time", "arguments": '{"location": "'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        ),
        # Frame 3: index 0 suffix (id and name omitted)
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
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": 'Paris"}'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        ),
        # Frame 4: index 1 suffix (id and name omitted)
        "data: "
        + json.dumps(
            {
                "id": "c4",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "glm-4",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 1,
                                    "function": {"arguments": 'Europe/Paris"}'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
        ),
        # Frame 5: normal finish_reason length as reporter observed on Vertex
        "data: "
        + json.dumps(
            {
                "id": "c5",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "glm-4",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
            }
        ),
        # Frame 6: empty-choice usage-only terminal completion_tokens=64
        "data: "
        + json.dumps(
            {
                "id": "c6",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "glm-4",
                "choices": [],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 64,
                    "total_tokens": 76,
                    "completion_tokens_details": {"reasoning_tokens": 18},
                },
            }
        ),
        "data: [DONE]",
    ]

    sse_bytes: Final[bytes] = ("\n\n".join(events) + "\n\n").encode("utf-8")

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse_bytes)

    mock_logging: Final = Mock()
    mock_logging.model_call_details = {"litellm_params": {}}
    mock_logging.stream_options = {"include_usage": True}

    raw_stream: Final = make_sync_call(
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond))),
        api_base="https://openai-like.example/v1/chat/completions",
        headers={},
        data=json.dumps({"model": "glm-4", "messages": []}),
        model="glm-4",
        messages=[],
        logging_obj=mock_logging,
        fake_stream=False,
    )
    wrapper: Final = CustomStreamWrapper(
        completion_stream=raw_stream,
        model="glm-4",
        custom_llm_provider="vertex_ai",
        logging_obj=mock_logging,
        stream_options={"include_usage": True},
    )

    chunks: Final = list(wrapper)
    assert len(chunks) == 6

    # Assert per-frame index/id/argument preservation
    # Frame 0: index 0 initial
    assert chunks[0].choices[0].delta.tool_calls[0].index == 0
    assert chunks[0].choices[0].delta.tool_calls[0].id == "call_weather"
    assert chunks[0].choices[0].delta.tool_calls[0].function.name == "get_weather"
    assert chunks[0].choices[0].delta.tool_calls[0].function.arguments == '{"city": "'

    # Frame 1: index 1 initial
    assert chunks[1].choices[0].delta.tool_calls[0].index == 1
    assert chunks[1].choices[0].delta.tool_calls[0].id == "call_time"
    assert chunks[1].choices[0].delta.tool_calls[0].function.name == "get_time"
    assert chunks[1].choices[0].delta.tool_calls[0].function.arguments == '{"location": "'

    # Frame 2: index 0 continuation (id/name absent)
    assert chunks[2].choices[0].delta.tool_calls[0].index == 0
    assert chunks[2].choices[0].delta.tool_calls[0].id is None
    assert chunks[2].choices[0].delta.tool_calls[0].function.name is None
    assert chunks[2].choices[0].delta.tool_calls[0].function.arguments == 'Paris"}'

    # Frame 3: index 1 continuation (id/name absent)
    assert chunks[3].choices[0].delta.tool_calls[0].index == 1
    assert chunks[3].choices[0].delta.tool_calls[0].id is None
    assert chunks[3].choices[0].delta.tool_calls[0].function.name is None
    assert chunks[3].choices[0].delta.tool_calls[0].function.arguments == 'Europe/Paris"}'

    # Frame 4: finish_reason length
    assert chunks[4].choices[0].finish_reason == "length"

    # Frame 5: usage chunk
    assert chunks[5].usage is not None
    assert chunks[5].usage.completion_tokens == 64
    assert chunks[5].usage.prompt_tokens == 12

    # Reassembling in client: final separately ordered complete arguments
    assembled_calls: Final[dict[int, dict[str, str]]] = {}
    for c in chunks:
        if c.choices and c.choices[0].delta and c.choices[0].delta.tool_calls:
            for tc in c.choices[0].delta.tool_calls:
                idx = tc.index
                if idx not in assembled_calls:
                    assembled_calls[idx] = {"id": tc.id or "", "name": tc.function.name or "", "arguments": ""}
                if tc.id:
                    assembled_calls[idx]["id"] = tc.id
                if tc.function.name:
                    assembled_calls[idx]["name"] = tc.function.name
                assembled_calls[idx]["arguments"] += tc.function.arguments

    assert assembled_calls[0]["id"] == "call_weather"
    assert assembled_calls[0]["name"] == "get_weather"
    assert assembled_calls[0]["arguments"] == '{"city": "Paris"}'

    assert assembled_calls[1]["id"] == "call_time"
    assert assembled_calls[1]["name"] == "get_time"
    assert assembled_calls[1]["arguments"] == '{"location": "Europe/Paris"}'


def test_real_stream_content_only_and_no_provider_usage_negative_controls():
    # Negative control 1: content-only stream emits no ghost tool calls, reasoning, or provider usage
    events_content: Final[list[str]] = [
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
                        "delta": {"content": "Direct text reply."},
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
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        ),
        "data: [DONE]",
    ]
    sse_bytes_content: Final[bytes] = ("\n\n".join(events_content) + "\n\n").encode("utf-8")

    def respond_content(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse_bytes_content)

    mock_logging_content: Final = Mock()
    mock_logging_content.model_call_details = {"litellm_params": {}}
    mock_logging_content.stream_options = {}

    stream_content: Final = make_sync_call(
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond_content))),
        api_base="https://openai-like.example/v1/chat/completions",
        headers={},
        data=json.dumps({"model": "glm-4", "messages": []}),
        model="glm-4",
        messages=[],
        logging_obj=mock_logging_content,
        fake_stream=False,
    )
    wrapper_content: Final = CustomStreamWrapper(
        completion_stream=stream_content,
        model="glm-4",
        custom_llm_provider="vertex_ai",
        logging_obj=mock_logging_content,
        stream_options={},
    )
    chunks_content: Final = list(wrapper_content)
    assert len(chunks_content) == 2
    assert chunks_content[0].choices[0].delta.content == "Direct text reply."
    assert chunks_content[0].choices[0].delta.tool_calls is None
    assert getattr(chunks_content[0].choices[0].delta, "reasoning_content", None) is None
    assert getattr(chunks_content[0], "usage", None) is None
    assert chunks_content[1].choices[0].finish_reason == "stop"

    # Negative control 2: no-provider-usage stream terminates cleanly without synthetic usage chunks
    events_no_usage: Final[list[str]] = [
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
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_single",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{}"},
                                }
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
                "id": "c2",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "glm-4",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            }
        ),
        "data: [DONE]",
    ]
    sse_bytes_no_usage: Final[bytes] = ("\n\n".join(events_no_usage) + "\n\n").encode("utf-8")

    def respond_no_usage(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse_bytes_no_usage)

    mock_logging_no_usage: Final = Mock()
    mock_logging_no_usage.model_call_details = {"litellm_params": {}}
    mock_logging_no_usage.stream_options = {}

    stream_no_usage: Final = make_sync_call(
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond_no_usage))),
        api_base="https://openai-like.example/v1/chat/completions",
        headers={},
        data=json.dumps({"model": "glm-4", "messages": []}),
        model="glm-4",
        messages=[],
        logging_obj=mock_logging_no_usage,
        fake_stream=False,
    )
    wrapper_no_usage: Final = CustomStreamWrapper(
        completion_stream=stream_no_usage,
        model="glm-4",
        custom_llm_provider="vertex_ai",
        logging_obj=mock_logging_no_usage,
        stream_options={},
    )
    chunks_no_usage: Final = list(wrapper_no_usage)
    assert len(chunks_no_usage) == 2
    assert chunks_no_usage[0].choices[0].delta.tool_calls[0].id == "call_single"
    assert getattr(chunks_no_usage[0], "usage", None) is None
    assert getattr(chunks_no_usage[1], "usage", None) is None
    assert chunks_no_usage[1].choices[0].finish_reason == "tool_calls"


def test_legacy_tool_use_fallback_covered():
    from litellm.litellm_core_utils.streaming_handler import convert_generic_chunk_to_model_response_stream
    from litellm.types.llms.openai import (
        ChatCompletionToolCallChunk,
        ChatCompletionToolCallFunctionChunk,
    )
    from litellm.types.utils import GenericStreamingChunk

    legacy_chunk: Final[GenericStreamingChunk] = GenericStreamingChunk(
        text="",
        tool_use=ChatCompletionToolCallChunk(
            id="call_legacy",
            type="function",
            function=ChatCompletionToolCallFunctionChunk(name="old_tool", arguments="{}"),
            index=0,
        ),
        is_finished=False,
        finish_reason="",
        usage=None,
        index=0,
    )
    res_stream: Final = convert_generic_chunk_to_model_response_stream(legacy_chunk)
    assert res_stream.choices[0].delta.tool_calls is not None
    assert res_stream.choices[0].delta.tool_calls[0].id == "call_legacy"
