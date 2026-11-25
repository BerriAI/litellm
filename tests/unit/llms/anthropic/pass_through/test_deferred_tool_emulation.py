import json
from collections.abc import AsyncIterator
from typing import Final

import pytest

from litellm.llms.anthropic.pass_through.deferred_tool_emulation import DeferredToolEmulation, deferred_tool_emulation
from litellm.llms.anthropic.pass_through.utils import anthropic_sse_frame
from litellm.responses.streaming_iterator import partial_responses_incomplete_event_on_disconnect
from litellm.types.llms.openai import ResponseIncompleteEvent, ResponsesAPIStreamEvents

_GREP_TOOL: Final = {"name": "Grep", "input_schema": {"type": "object"}, "defer_loading": True}
_DELETE_TOOL: Final = {"name": "Delete", "input_schema": {"type": "object"}, "defer_loading": True}
_TOOL_SEARCH: Final = {"name": "ToolSearch", "input_schema": {"type": "object"}}
_EMULATE: Final = {"deferred_tools": "emulate"}
_EMULATION: Final = DeferredToolEmulation(deferred={"Grep": _GREP_TOOL}, loaded=frozenset({"Grep"}))


def _tool_search_history(*loaded: str) -> list[dict[str, object]]:
    return [
        {"role": "user", "content": "Find the tools you need"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "toolu_0", "name": "ToolSearch", "input": {"query": "tools"}}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_0",
                    "content": [{"type": "tool_reference", "tool_name": name} for name in loaded],
                }
            ],
        },
    ]


def _emulation(
    tools: list[dict[str, object]], tool_choice: dict[str, object] | None, *loaded: str
) -> DeferredToolEmulation:
    emulation: Final = deferred_tool_emulation(
        _EMULATE, tools=tools, tool_choice=tool_choice, messages=_tool_search_history(*loaded)
    )
    assert emulation is not None
    return emulation


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


class _SyncStreamThatAlsoIteratesAsync:
    def __init__(self, frames: tuple[bytes, ...]) -> None:
        self._frames: Final = iter(frames)

    def __iter__(self) -> "_SyncStreamThatAlsoIteratesAsync":
        return self

    def __next__(self) -> bytes:
        return next(self._frames)

    def __aiter__(self) -> "_SyncStreamThatAlsoIteratesAsync":
        return self

    async def __anext__(self) -> bytes:
        raise StopAsyncIteration


def test_a_sync_stream_that_also_iterates_async_restores_the_dispatcher_call_when_read_sync() -> None:
    frames: Final = _dispatcher_stream_frames(json.dumps({"name": "Grep", "input": {"pattern": "TODO"}}))

    streamed: Final = b"".join(_EMULATION.restored(_SyncStreamThatAlsoIteratesAsync(frames)))

    assert b"CallLoadedTool" not in streamed
    assert b'"name": "Grep"' in streamed
    assert streamed.endswith(anthropic_sse_frame({"type": "message_stop"}))


@pytest.mark.parametrize(
    "tool_choice, sent",
    [
        ({"type": "tool", "name": "Grep"}, {"type": "tool", "name": "CallLoadedTool"}),
        ({"type": "tool", "name": "Read"}, {"type": "tool", "name": "Read"}),
        ({"type": "any"}, {"type": "any"}),
        (None, None),
    ],
    ids=["forced_deferred_tool", "forced_loaded_tool", "any", "unset"],
)
def test_a_forced_deferred_tool_is_forced_through_the_dispatcher(
    tool_choice: dict[str, object] | None, sent: dict[str, object] | None
) -> None:
    assert _emulation([_TOOL_SEARCH, _GREP_TOOL], tool_choice, "Grep").tool_choice(tool_choice) == sent


def _dispatcher_response(dispatcher: str, tool_name: str) -> dict[str, object]:
    return {
        "content": [
            {"type": "tool_use", "id": "toolu_1", "name": dispatcher, "input": {"name": tool_name, "input": {}}},
        ]
    }


def test_a_forced_deferred_tool_cannot_be_swapped_for_another_deferred_tool() -> None:
    tools: Final = [_TOOL_SEARCH, _GREP_TOOL, _DELETE_TOOL]
    emulation: Final = _emulation(tools, {"type": "tool", "name": "Grep"}, "Grep", "Delete")
    dispatcher_tool: Final = emulation.tools(tools)[-1]
    swapped: Final = _dispatcher_response("CallLoadedTool", "Delete")

    assert dispatcher_tool["input_schema"]["properties"]["name"]["enum"] == ["Grep"]
    assert emulation.restored(swapped) == swapped
    assert emulation.restored(_dispatcher_response("CallLoadedTool", "Grep"))["content"][0]["name"] == "Grep"


def test_the_dispatcher_never_takes_the_name_of_a_tool_the_caller_sent() -> None:
    own_tool: Final = {"name": "CallLoadedTool", "input_schema": {"type": "object"}}
    tools: Final = [_TOOL_SEARCH, own_tool, _GREP_TOOL]
    emulation: Final = _emulation(tools, None, "Grep")
    sent_names: Final = [tool["name"] for tool in emulation.tools(tools)]
    own_call: Final = _dispatcher_response("CallLoadedTool", "Grep")
    dispatcher: Final = sent_names[-1]

    assert len(sent_names) == len(set(sent_names))
    assert emulation.restored(own_call) == own_call
    assert emulation.restored(_dispatcher_response(dispatcher, "Grep"))["content"][0]["name"] == "Grep"


def test_a_dispatcher_call_to_a_deferred_tool_tool_search_never_loaded_is_kept_as_sent() -> None:
    emulation: Final = _emulation([_TOOL_SEARCH, _GREP_TOOL, _DELETE_TOOL], None, "Grep")
    unloaded: Final = _dispatcher_response("CallLoadedTool", "Delete")

    assert emulation.restored(unloaded) == unloaded
    assert emulation.restored(_dispatcher_response("CallLoadedTool", "Grep"))["content"][0]["name"] == "Grep"


class _ClosableSyncStream:
    def __init__(self, frames: tuple[bytes, ...]) -> None:
        self._frames: Final = iter(frames)
        self.closed = False

    def __iter__(self) -> "_ClosableSyncStream":
        return self

    def __next__(self) -> bytes:
        return next(self._frames)

    def close(self) -> None:
        self.closed = True


class _ResponsesBridgeStream:
    def __init__(self, frames: tuple[bytes, ...], partial: ResponseIncompleteEvent) -> None:
        self._frames: Final = iter(frames)
        self._partial: Final = partial

    def __aiter__(self) -> "_ResponsesBridgeStream":
        return self

    async def __anext__(self) -> bytes:
        try:
            return next(self._frames)
        except StopIteration:
            raise StopAsyncIteration from None

    def partial_response_on_disconnect(self) -> ResponseIncompleteEvent:
        return self._partial


@pytest.mark.asyncio
async def test_a_restored_stream_the_client_drops_still_bills_what_the_provider_streamed() -> None:
    partial: Final = ResponseIncompleteEvent.model_construct(type=ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE)
    upstream: Final = _ResponsesBridgeStream(_dispatcher_stream_frames(json.dumps({"name": "Grep", "input": {}})), partial)
    restored: Final = _EMULATION.restored(upstream)
    await restored.__anext__()

    assert partial_responses_incomplete_event_on_disconnect(restored) is partial


@pytest.mark.parametrize("frames_read", [0, 1], ids=["before_reading", "mid_stream"])
def test_closing_a_restored_sync_stream_closes_the_provider_stream(frames_read: int) -> None:
    upstream: Final = _ClosableSyncStream(_dispatcher_stream_frames(json.dumps({"name": "Grep", "input": {}})))
    restored: Final = _EMULATION.restored(upstream)
    for _ in range(frames_read):
        next(restored)

    restored.close()

    assert upstream.closed


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
