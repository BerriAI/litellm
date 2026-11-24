import json
from collections.abc import AsyncIterator
from typing import Final

import pytest

from litellm.llms.anthropic.pass_through.deferred_tool_emulation import DeferredToolEmulation
from litellm.llms.anthropic.pass_through.utils import anthropic_sse_frame

_GREP_TOOL: Final = {"name": "Grep", "input_schema": {"type": "object"}, "defer_loading": True}
_EMULATION: Final = DeferredToolEmulation(deferred={"Grep": _GREP_TOOL})


def _dispatcher_stream_frames(dispatcher_input: str) -> tuple[bytes, ...]:
    return (
        anthropic_sse_frame({"type": "message_start", "message": {"id": "msg_1", "content": []}}),
        anthropic_sse_frame(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "CallLoadedTool", "input": {}},
            }
        ),
        anthropic_sse_frame(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": dispatcher_input},
            }
        ),
        anthropic_sse_frame({"type": "content_block_stop", "index": 0}),
        anthropic_sse_frame({"type": "message_stop"}),
    )


async def _split_mid_frame(frames: tuple[bytes, ...]) -> AsyncIterator[bytes]:
    joined: Final = b"".join(frames)
    for start in range(0, len(joined), 7):
        yield joined[start : start + 7]


async def _restored_stream(frames: tuple[bytes, ...]) -> bytes:
    restored: Final = _EMULATION.restored(_split_mid_frame(frames))
    return b"".join([frame async for frame in restored])


@pytest.mark.asyncio
async def test_a_stream_restores_a_dispatcher_call_split_across_chunks() -> None:
    streamed: Final = await _restored_stream(
        _dispatcher_stream_frames(json.dumps({"name": "Grep", "input": {"pattern": "TODO"}}))
    )

    assert b"CallLoadedTool" not in streamed
    assert b'"name": "Grep"' in streamed
    assert b'"partial_json": "{\\"pattern\\": \\"TODO\\"}"' in streamed
    assert streamed.endswith(anthropic_sse_frame({"type": "message_stop"}))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dispatcher_input",
    [{"input": {"pattern": "TODO"}}, {"name": "Bash", "input": {"command": "ls"}}],
    ids=["no_tool_name", "tool_the_request_did_not_defer"],
)
async def test_a_stream_keeps_a_dispatcher_call_it_cannot_restore_as_sent(dispatcher_input: dict[str, object]) -> None:
    frames: Final = _dispatcher_stream_frames(json.dumps(dispatcher_input))

    assert await _restored_stream(frames) == b"".join(frames)


@pytest.mark.parametrize(
    "dispatcher_input",
    [{"name": "Grep", "input": {"pattern": "TODO"}}, {"name": "Grep", "input": '{"pattern": "TODO"}'}],
    ids=["object_input", "json_string_input"],
)
def test_a_response_restores_the_loaded_tool_call(dispatcher_input: dict[str, object]) -> None:
    response: Final = {
        "content": [
            {"type": "text", "text": "Searching"},
            {"type": "tool_use", "id": "toolu_1", "name": "CallLoadedTool", "input": dispatcher_input},
        ]
    }

    assert _EMULATION.restored(response)["content"] == [
        {"type": "text", "text": "Searching"},
        {"type": "tool_use", "id": "toolu_1", "name": "Grep", "input": {"pattern": "TODO"}},
    ]


def test_a_response_keeps_a_dispatcher_call_to_a_tool_the_request_did_not_defer() -> None:
    response: Final = {
        "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "CallLoadedTool", "input": {"name": "Bash", "input": {}}},
        ]
    }

    assert _EMULATION.restored(response) == response
