import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import CallToolResult, TextContent

import litellm  # noqa: F401 - ensures litellm.responses.main is registered in sys.modules
from litellm.responses.mcp.mcp_streaming_iterator import (
    MAX_MCP_TOOL_CALL_ROUNDS,
    MCPEnhancedStreamingIterator,
)
from litellm.types.llms.openai import ResponsesAPIResponse, ResponsesAPIStreamEvents

# `litellm.__init__` re-exports a function named `responses`, which shadows the
# `litellm.responses` subpackage as an attribute — `import litellm.responses.main`
# can resolve to the unrelated third-party `responses` package instead. Look the
# real submodule up in sys.modules directly to sidestep the shadowing.
responses_main_module = sys.modules["litellm.responses.main"]


class _FakeAsyncStream:
    """Minimal async iterator yielding pre-built chunks, one per __anext__ call."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


def _output_item_added_chunk():
    return SimpleNamespace(type=ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED)


def _completed_chunk(output):
    response = ResponsesAPIResponse(id="resp-1", created_at=0, output=output)
    return SimpleNamespace(type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED, response=response)


def _function_call(call_id: str, name: str, arguments: str = "{}"):
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments}


def _text_message(text: str):
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def _tool_call_stream(call_id: str, tool_name: str) -> _FakeAsyncStream:
    return _FakeAsyncStream([_completed_chunk([_function_call(call_id, tool_name)])])


def _text_only_stream(text: str) -> _FakeAsyncStream:
    return _FakeAsyncStream([_completed_chunk([_text_message(text)])])


def _mock_mcp_environment(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Patch the MCP tool-call plumbing so _execute_tool_calls can run in tests."""
    call_tool = AsyncMock(return_value=CallToolResult(content=[TextContent(type="text", text="ok")], isError=False))
    fake_manager = types.SimpleNamespace(
        call_tool=call_tool,
        _get_mcp_server_from_tool_name=MagicMock(return_value=None),
    )
    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
        fake_manager,
    )
    monkeypatch.setitem(
        sys.modules,
        "litellm.proxy.proxy_server",
        types.SimpleNamespace(proxy_logging_obj=MagicMock()),
    )
    return call_tool


def _make_iterator(initial_chunks) -> MCPEnhancedStreamingIterator:
    return MCPEnhancedStreamingIterator(
        base_iterator=_FakeAsyncStream(initial_chunks),
        mcp_events=[],
        tool_server_map={"read_wiki_contents": "deepwiki"},
        mcp_tools_with_litellm_proxy=[{"require_approval": "never"}],
        user_api_key_auth=None,
        original_request_params={
            "model": "gpt-4",
            "input": "what is berriai/litellm?",
            "tools": [{"type": "mcp"}],
        },
    )


@pytest.mark.asyncio
async def test_second_round_tool_call_is_executed_and_reaches_final_text(monkeypatch):
    """
    Regression test: a tool call that errors, followed by the model retrying
    the tool in its follow-up turn, must have that second tool call executed
    too — the stream should not silently end after round 1 with no final
    text (see PR discussion: deepwiki tool call errors, model retries once,
    reply used to just stop with no explanation).
    """
    call_tool = _mock_mcp_environment(monkeypatch)

    aresponses_mock = AsyncMock(
        side_effect=[
            _tool_call_stream("call_2", "read_wiki_contents"),  # round 2: model retries
            _text_only_stream("Here's what I found after retrying."),  # round 3: final answer
        ]
    )
    monkeypatch.setattr(responses_main_module, "aresponses", aresponses_mock)

    iterator = _make_iterator(
        [
            _output_item_added_chunk(),
            _completed_chunk([_function_call("call_1", "read_wiki_contents")]),  # round 1: errors
        ]
    )

    chunks = [chunk async for chunk in iterator]

    # Both rounds' tool calls were actually executed, not just streamed unexecuted.
    assert call_tool.call_count == 2
    assert iterator.tool_call_round == 2

    # The stream reached round 3 and produced the final text response instead
    # of stopping after round 1 or round 2.
    completed_chunks = [c for c in chunks if getattr(c, "type", None) == ResponsesAPIStreamEvents.RESPONSE_COMPLETED]
    assert len(completed_chunks) == 3
    final_output = completed_chunks[-1].response.output
    assert final_output[0]["content"][0]["text"] == "Here's what I found after retrying."


@pytest.mark.asyncio
async def test_tool_call_rounds_are_capped(monkeypatch):
    """
    A model that keeps calling tools every round must not loop forever —
    auto-execution stops at MAX_MCP_TOOL_CALL_ROUNDS, and the follow-up made
    at the cap drops "tools" from the request so the model is forced to
    answer in text instead of the stream just hanging.
    """
    call_tool = _mock_mcp_environment(monkeypatch)

    # Rounds 2..MAX_MCP_TOOL_CALL_ROUNDS keep calling the tool; the call made
    # once the cap is hit returns a text-only response.
    tool_call_streams = [
        _tool_call_stream(f"call_{i}", "read_wiki_contents") for i in range(2, MAX_MCP_TOOL_CALL_ROUNDS + 1)
    ]
    aresponses_mock = AsyncMock(side_effect=[*tool_call_streams, _text_only_stream("giving up on tools")])
    monkeypatch.setattr(responses_main_module, "aresponses", aresponses_mock)

    iterator = _make_iterator(
        [
            _output_item_added_chunk(),
            _completed_chunk([_function_call("call_1", "read_wiki_contents")]),
        ]
    )

    _ = [chunk async for chunk in iterator]

    assert iterator.tool_call_round == MAX_MCP_TOOL_CALL_ROUNDS
    assert call_tool.call_count == MAX_MCP_TOOL_CALL_ROUNDS
    assert aresponses_mock.call_count == MAX_MCP_TOOL_CALL_ROUNDS

    # Every follow-up before the cap still offered tools; only the capped one drops them.
    for call in aresponses_mock.call_args_list[:-1]:
        assert "tools" in call.kwargs
    assert "tools" not in aresponses_mock.call_args_list[-1].kwargs
