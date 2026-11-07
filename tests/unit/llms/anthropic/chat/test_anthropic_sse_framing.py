import json

import pytest

import litellm
from litellm.llms.anthropic.chat.handler import ModelResponseIterator


def test_anthropic_streaming_thinking_text_tool_use_sync():
    sse_lines = [
        b"event: message_start\n",
        b'data: {"type":"message_start","message":{"id":"msg_test_123","type":"message","role":"assistant","model":"claude-3-7-sonnet-20250219","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}\n',
        b"\n",
        b"event: content_block_start\n",
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}\n',
        b"\n",
        b"event: content_block_delta\n",
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me call submit_result"}}\n',
        b"\n",
        b"event: content_block_stop\n",
        b'data: {"type":"content_block_stop","index":0}\n',
        b"\n",
        b"event: content_block_start\n",
        b'data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}\n',
        b"\n",
        b"event: content_block_delta\n",
        b'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"I will now submit the result."}}\n',
        b"\n",
        b"event: content_block_stop\n",
        b'data: {"type":"content_block_stop","index":1}\n',
        b"\n",
        b"event: content_block_start\n",
        b'data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_01","name":"submit_result","input":{}}}\n',
        b"\n",
        b"event: content_block_delta\n",
        b'data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\\"value\\":"}}\n',
        b"\n",
        b"event: content_block_delta\n",
        b'data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"1}"}}\n',
        b"\n",
        b"event: content_block_stop\n",
        b'data: {"type":"content_block_stop","index":2}\n',
        b"\n",
        b"event: message_delta\n",
        b'data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":25}}\n',
        b"\n",
        b"event: message_stop\n",
        b'data: {"type":"message_stop"}\n',
    ]

    iterator = ModelResponseIterator(
        streaming_response=iter(sse_lines),
        sync_stream=True,
        json_mode=False,
    )
    chunks = list(iterator)

    assert all(isinstance(c, litellm.ModelResponseStream) for c in chunks)

    tool_call_chunks = [c for c in chunks if getattr(c.choices[0].delta, "tool_calls", None) is not None]
    assert len(tool_call_chunks) == 3
    assert tool_call_chunks[0].choices[0].delta.tool_calls[0].function.name == "submit_result"
    assert tool_call_chunks[0].choices[0].delta.tool_calls[0].id == "toolu_01"

    built = litellm.stream_chunk_builder(chunks=chunks)
    assert built.choices[0].message.content == "I will now submit the result."
    assert built.choices[0].message.tool_calls is not None
    assert len(built.choices[0].message.tool_calls) == 1
    assert built.choices[0].message.tool_calls[0].function.name == "submit_result"
    assert built.choices[0].message.tool_calls[0].function.arguments == '{"value":1}'
    assert built.choices[0].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_anthropic_streaming_thinking_text_tool_use_async():
    sse_lines = [
        "event: message_start\n",
        'data: {"type":"message_start","message":{"id":"msg_async_123","type":"message","role":"assistant","model":"claude-3-7-sonnet-20250219","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":1}}}\n',
        "event: content_block_start\n",
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}\n',
        "event: content_block_delta\n",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Async thinking..."}}\n',
        "event: content_block_stop\n",
        'data: {"type":"content_block_stop","index":0}\n',
        "event: content_block_start\n",
        'data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_async_01","name":"do_action","input":{}}}\n',
        "event: content_block_delta\n",
        'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"action\\": \\"run\\"}"}}\n',
        "event: content_block_stop\n",
        'data: {"type":"content_block_stop","index":1}\n',
        "event: message_delta\n",
        'data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":15}}\n',
        "event: message_stop\n",
        'data: {"type":"message_stop"}\n',
    ]

    class AsyncLineIterator:
        def __init__(self, lines):
            self._iter = iter(lines)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._iter)
            except StopIteration:
                raise StopAsyncIteration

    iterator = ModelResponseIterator(
        streaming_response=AsyncLineIterator(sse_lines),
        sync_stream=False,
        json_mode=False,
    )

    chunks = []
    async for chunk in iterator:
        chunks.append(chunk)

    assert all(isinstance(c, litellm.ModelResponseStream) for c in chunks)
    tool_call_chunks = [c for c in chunks if getattr(c.choices[0].delta, "tool_calls", None) is not None]
    assert len(tool_call_chunks) == 2
    assert tool_call_chunks[0].choices[0].delta.tool_calls[0].function.name == "do_action"

    built = litellm.stream_chunk_builder(chunks=chunks)
    assert built.choices[0].message.tool_calls[0].function.name == "do_action"
    assert json.loads(built.choices[0].message.tool_calls[0].function.arguments) == {"action": "run"}
