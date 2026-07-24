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


def _created_chunk(response_id: str):
    response = ResponsesAPIResponse(id=response_id, created_at=0, output=[])
    return SimpleNamespace(type=ResponsesAPIStreamEvents.RESPONSE_CREATED, response=response)


def _completed_chunk(output, response_id: str = "resp-1"):
    response = ResponsesAPIResponse(id=response_id, created_at=0, output=output)
    return SimpleNamespace(type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED, response=response)


def _function_call(call_id: str, name: str, arguments: str = "{}"):
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments}


def _text_message(text: str):
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def _tool_call_stream(call_id: str, tool_name: str, response_id: str = "resp-1") -> _FakeAsyncStream:
    return _FakeAsyncStream([_completed_chunk([_function_call(call_id, tool_name)], response_id=response_id)])


def _text_only_stream(text: str, response_id: str = "resp-1") -> _FakeAsyncStream:
    return _FakeAsyncStream([_completed_chunk([_text_message(text)], response_id=response_id)])


def _mock_mcp_environment(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Patch the MCP tool-call plumbing so _execute_tool_calls can run in tests."""
    call_tool = AsyncMock(return_value=CallToolResult(content=[TextContent(type="text", text="ok")], isError=False))
    fake_manager = types.SimpleNamespace(
        call_tool=call_tool,
        _get_mcp_server_from_tool_name=MagicMock(return_value=None),
        get_mcp_server_by_name=MagicMock(return_value=None),
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


@pytest.mark.asyncio
async def test_continuation_id_is_final_round_not_interim_tool_call(monkeypatch):
    """
    Regression test for the broken `previous_response_id` continuation after a
    gateway-executed MCP tool call. Each auto-execute round is a distinct
    upstream response: the interim round holds only the model's function_call
    (no tool output), the final round holds the answer. The client must be
    handed the FINAL round's response id, because that is the one whose stored
    chain includes the function_call_output. Pinning every event to the interim
    round's id made the next turn continue from a response with a dangling
    function_call, which the provider rejects with
    "No tool output found for function call ...".
    """
    _mock_mcp_environment(monkeypatch)

    aresponses_mock = AsyncMock(side_effect=[_text_only_stream("The first item is Alpha.", response_id="resp-final")])
    monkeypatch.setattr(responses_main_module, "aresponses", aresponses_mock)

    iterator = _make_iterator(
        [
            _created_chunk("resp-interim"),
            _output_item_added_chunk(),
            _completed_chunk([_function_call("call_1", "read_wiki_contents")], response_id="resp-interim"),
        ]
    )

    chunks = [chunk async for chunk in iterator]
    completed = [c for c in chunks if getattr(c, "type", None) == ResponsesAPIStreamEvents.RESPONSE_COMPLETED]

    assert completed[-1].response.output[0]["content"][0]["text"] == "The first item is Alpha."
    assert completed[-1].response.id == "resp-final"
    assert completed[-1].response.id != "resp-interim"


@pytest.mark.asyncio
async def test_follow_up_call_failure_emits_terminal_error_event(monkeypatch):
    """
    Regression test for the silent-swallow path: when the follow-up LLM call
    raises, the stream must emit a terminal `error` event instead of ending
    silently after the tool events (which surfaced to clients as a successful
    but empty completion).
    """
    _mock_mcp_environment(monkeypatch)

    aresponses_mock = AsyncMock(side_effect=RuntimeError("boom from provider"))
    monkeypatch.setattr(responses_main_module, "aresponses", aresponses_mock)

    iterator = _make_iterator(
        [
            _output_item_added_chunk(),
            _completed_chunk([_function_call("call_1", "read_wiki_contents")]),
        ]
    )

    chunks = [chunk async for chunk in iterator]
    error_events = [c for c in chunks if getattr(c, "type", None) == ResponsesAPIStreamEvents.ERROR]

    assert len(error_events) == 1
    assert error_events[0].error.type == "mcp_gateway_error"
    assert "boom from provider" in error_events[0].error.message


@pytest.mark.asyncio
async def test_initial_call_failure_is_stashed_for_eager_reraise(monkeypatch):
    """
    Regression test: a failing initial LLM call must be stashed as
    `_initial_creation_error` so `aresponses_api_with_mcp` can re-raise it as a
    real 4xx before any SSE bytes are written, instead of returning HTTP 200
    with an empty stream.
    """
    _mock_mcp_environment(monkeypatch)

    aresponses_mock = AsyncMock(side_effect=RuntimeError("initial boom"))
    monkeypatch.setattr(responses_main_module, "aresponses", aresponses_mock)

    iterator = _make_iterator([_output_item_added_chunk()])
    await iterator._create_initial_response_iterator()

    assert iterator._initial_creation_error is not None
    assert "initial boom" in str(iterator._initial_creation_error)
