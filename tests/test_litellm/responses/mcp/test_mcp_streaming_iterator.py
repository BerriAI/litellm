import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import litellm
from litellm.responses.mcp.mcp_streaming_iterator import MCPEnhancedStreamingIterator
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


def _completed_chunk(output):
    response = ResponsesAPIResponse(id="resp-1", created_at=0, output=output)
    return SimpleNamespace(type=ResponsesAPIStreamEvents.RESPONSE_COMPLETED, response=response)


def _text_message(text: str):
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def _text_only_stream(text: str) -> _FakeAsyncStream:
    return _FakeAsyncStream([_completed_chunk([_text_message(text)])])


def _make_lazy_iterator(mcp_events=None) -> MCPEnhancedStreamingIterator:
    """Iterator with no base_iterator: the initial LLM call happens lazily on iteration."""
    return MCPEnhancedStreamingIterator(
        base_iterator=None,
        mcp_events=list(mcp_events or []),
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
async def test_initial_call_failure_emits_error_event_not_discovery_events(monkeypatch):
    """
    Regression test: when the initial LLM call fails (e.g. an invalid
    previous_response_id -> provider 400 "No tool output found for function
    call ..."), the stream used to emit the pre-generated mcp_list_tools
    discovery events with no response.created before them — which violates
    the Responses API streaming contract and crashes SDK stream accumulators
    (openai-node: "expected 'response.created' event, got
    response.mcp_list_tools.in_progress"). The stream must instead surface a
    single terminal `error` event and end.
    """
    aresponses_mock = AsyncMock(
        side_effect=litellm.BadRequestError(
            message="No tool output found for function call call_x.",
            model="gpt-4",
            llm_provider="openai",
        )
    )
    monkeypatch.setattr(responses_main_module, "aresponses", aresponses_mock)

    discovery_event = SimpleNamespace(type=ResponsesAPIStreamEvents.MCP_LIST_TOOLS_IN_PROGRESS)
    iterator = _make_lazy_iterator(mcp_events=[discovery_event])

    chunks = [chunk async for chunk in iterator]

    assert len(chunks) == 1
    error_event = chunks[0]
    assert error_event.type == ResponsesAPIStreamEvents.ERROR
    assert error_event.error.code == "400"
    assert "No tool output found" in error_event.error.message
    # No discovery events leaked before/after the error.
    assert discovery_event not in chunks


@pytest.mark.asyncio
async def test_eager_creation_reraises_pre_stream_failure_as_http_error(monkeypatch):
    """
    aresponses_api_with_mcp creates the initial response eagerly and re-raises
    the stashed creation failure, so the proxy returns a real 4xx/5xx before
    any SSE bytes are written instead of an HTTP 200 with a broken stream.
    """
    from litellm.responses.mcp.litellm_proxy_mcp_handler import LiteLLM_Proxy_MCP_Handler

    monkeypatch.setattr(
        LiteLLM_Proxy_MCP_Handler,
        "_process_mcp_tools_without_openai_transform",
        AsyncMock(return_value=([], {})),
    )
    boom = litellm.BadRequestError(
        message="Previous response with id 'resp_bogus' not found.",
        model="gpt-4",
        llm_provider="openai",
    )
    monkeypatch.setattr(responses_main_module, "aresponses", AsyncMock(side_effect=boom))

    with pytest.raises(litellm.BadRequestError) as excinfo:
        await responses_main_module.aresponses_api_with_mcp(
            input="hi",
            model="gpt-4",
            stream=True,
            previous_response_id="resp_bogus",
            tools=[{"type": "mcp", "server_url": "litellm_proxy/mcp/deepwiki", "require_approval": "never"}],
        )

    assert "resp_bogus" in str(excinfo.value)


@pytest.mark.asyncio
async def test_initial_call_success_does_not_emit_error_event(monkeypatch):
    """Happy path is unchanged: no error event, stream flows as before."""
    monkeypatch.setattr(
        responses_main_module,
        "aresponses",
        AsyncMock(return_value=_text_only_stream("all good")),
    )

    iterator = _make_lazy_iterator()
    chunks = [chunk async for chunk in iterator]

    assert all(getattr(c, "type", None) != ResponsesAPIStreamEvents.ERROR for c in chunks)
    completed = [c for c in chunks if getattr(c, "type", None) == ResponsesAPIStreamEvents.RESPONSE_COMPLETED]
    assert len(completed) == 1
    assert iterator._initial_creation_error is None
