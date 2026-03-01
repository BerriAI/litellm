"""
Tests for x-litellm-trace-id propagation across LLM, MCP, and A2A calls.

Verifies that when a caller sends x-litellm-trace-id, the trace_id flows through:
  1. LLM calls: proxy extracts header → data["litellm_trace_id"] → StandardLoggingPayload.trace_id
  2. A2A completion bridge: metadata["trace_id"] → litellm.acompletion(litellm_trace_id=...)
  3. A2A standard flow: metadata["trace_id"] → X-LiteLLM-Trace-Id header to downstream agent

Run with:
    pytest tests/test_litellm/test_a2a_trace_id_propagation.py -v
"""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_a2a_params(text: str = "Hello") -> dict:
    return {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": text}],
            "messageId": uuid4().hex,
        }
    }


def _mock_completion_response():
    """Minimal object that looks like a litellm ModelResponse."""
    choice = MagicMock()
    choice.message = MagicMock(content="Hi there")
    resp = MagicMock()
    resp.choices = [choice]
    resp.id = "chatcmpl-mock"
    return resp


async def _mock_streaming_response():
    """Async generator yielding one chunk."""
    chunk = MagicMock()
    chunk.choices = [MagicMock(delta=MagicMock(content="Hi"))]
    yield chunk


# ===========================================================================
# 1. LLM CALLS — proxy header extraction → litellm_trace_id on the data dict
# ===========================================================================


def test_proxy_extracts_trace_id_header_to_top_level_kwarg():
    """
    When the proxy processes a request with x-litellm-trace-id header,
    it should set data["litellm_trace_id"] so the router/completion uses it
    instead of generating a random UUID.

    Today this FAILS: the header only goes to metadata["trace_id"],
    never to the top-level litellm_trace_id kwarg.
    """
    from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup

    trace_id = f"trace-{uuid4().hex[:8]}"

    # Simulate incoming request data as the proxy would build it
    data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hi"}],
        "metadata": {},
    }
    headers = {"x-litellm-trace-id": trace_id}

    # Call the static method that processes headers
    result = LiteLLMProxyRequestSetup.add_litellm_metadata_from_request_headers(
        headers=headers,
        data=data,
        _metadata_variable_name="metadata",
    )

    # The trace_id should be in metadata (this already works)
    assert result["metadata"].get("trace_id") == trace_id

    # AND it should be set as a top-level kwarg so the router uses it
    assert result.get("litellm_trace_id") == trace_id, (
        f"Expected data['litellm_trace_id']={trace_id!r}, "
        f"but it was not set. The header value only went to metadata['trace_id']. "
        f"The router will generate a random UUID instead."
    )


def test_logging_payload_resolves_trace_id_from_metadata():
    """
    When litellm_params has metadata with trace_id but no top-level
    litellm_trace_id, the logging system should still use it.

    This tests the @client decorator path (A2A calls) where metadata
    is set but litellm_trace_id is not extracted to litellm_params.
    """
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    trace_id = f"trace-{uuid4().hex[:8]}"

    mock_logging_obj = MagicMock()
    mock_logging_obj.litellm_trace_id = "auto-generated-fallback"

    # Simulate what function_setup does for @client decorated calls:
    # metadata goes into litellm_params["metadata"] but trace_id is NOT
    # extracted to litellm_params["litellm_trace_id"]
    litellm_params = {
        "api_base": "",
        "metadata": {"trace_id": trace_id},
        # litellm_trace_id is NOT set — this is the gap
    }

    result = StandardLoggingPayloadSetup._get_standard_logging_payload_trace_id(
        logging_obj=mock_logging_obj,
        litellm_params=litellm_params,
    )

    assert result == trace_id, (
        f"Expected trace_id={trace_id!r} from metadata, "
        f"got {result!r}. The logging system ignores metadata['trace_id'] "
        f"and falls back to the auto-generated ID."
    )


# ===========================================================================
# 2. A2A — completion bridge (non-streaming)
# ===========================================================================


@pytest.mark.asyncio
async def test_completion_bridge_non_streaming_forwards_trace_id():
    """
    When handle_non_streaming receives metadata={"trace_id": "abc"},
    it should pass litellm_trace_id="abc" to litellm.acompletion.
    """
    from litellm.a2a_protocol.litellm_completion_bridge.handler import (
        A2ACompletionBridgeHandler,
    )

    captured = {}

    async def mock_acompletion(**kwargs):
        captured.update(kwargs)
        return _mock_completion_response()

    trace_id = f"trace-{uuid4().hex[:8]}"

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        await A2ACompletionBridgeHandler.handle_non_streaming(
            request_id=str(uuid4()),
            params=_make_a2a_params(),
            litellm_params={"custom_llm_provider": "openai", "model": "gpt-4"},
            api_base="http://localhost:4000",
            metadata={"trace_id": trace_id},
        )

    assert captured.get("litellm_trace_id") == trace_id, (
        f"Expected litellm_trace_id={trace_id!r}, got {captured.get('litellm_trace_id')!r}"
    )


# ===========================================================================
# 3. A2A — completion bridge (streaming)
# ===========================================================================


@pytest.mark.asyncio
async def test_completion_bridge_streaming_forwards_trace_id():
    """
    When handle_streaming receives metadata={"trace_id": "abc"},
    it should pass litellm_trace_id="abc" to litellm.acompletion.
    """
    from litellm.a2a_protocol.litellm_completion_bridge.handler import (
        A2ACompletionBridgeHandler,
    )

    captured = {}

    async def mock_acompletion(**kwargs):
        captured.update(kwargs)
        return _mock_streaming_response()

    trace_id = f"trace-{uuid4().hex[:8]}"

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        chunks = []
        async for chunk in A2ACompletionBridgeHandler.handle_streaming(
            request_id=str(uuid4()),
            params=_make_a2a_params(),
            litellm_params={"custom_llm_provider": "openai", "model": "gpt-4"},
            api_base="http://localhost:4000",
            metadata={"trace_id": trace_id},
        ):
            chunks.append(chunk)

    assert captured.get("litellm_trace_id") == trace_id, (
        f"Expected litellm_trace_id={trace_id!r}, got {captured.get('litellm_trace_id')!r}"
    )


# ===========================================================================
# 4. A2A — standard flow reuses caller's trace_id
# ===========================================================================


@pytest.mark.asyncio
async def test_standard_a2a_flow_reuses_callers_trace_id():
    """
    When asend_message is called WITHOUT custom_llm_provider (standard A2A flow)
    and metadata contains trace_id, the X-LiteLLM-Trace-Id header sent to the
    downstream agent should be the caller's trace_id, not a new random uuid.
    """
    captured_headers = {}

    async def mock_create_a2a_client(base_url, extra_headers=None, **kwargs):
        if extra_headers:
            captured_headers.update(extra_headers)
        mock_client = AsyncMock()
        mock_client._litellm_agent_card = MagicMock(name="test-agent", url=base_url)
        mock_client.agent_card = mock_client._litellm_agent_card
        mock_response = MagicMock()
        mock_response.root = MagicMock()
        mock_response.root.result = MagicMock()
        mock_client.send_message = AsyncMock(return_value=mock_response)
        return mock_client

    trace_id = f"trace-{uuid4().hex[:8]}"

    from a2a.types import MessageSendParams, SendMessageRequest

    from litellm.a2a_protocol.main import asend_message

    request = SendMessageRequest(
        id=str(uuid4()),
        params=MessageSendParams(
            message={
                "role": "user",
                "parts": [{"kind": "text", "text": "Hello"}],
                "messageId": uuid4().hex,
            }
        ),
    )

    mock_logging_obj = MagicMock()
    mock_logging_obj.async_failure_handler = AsyncMock()
    mock_logging_obj.async_success_handler = AsyncMock()

    def mock_function_setup(*args, **kwargs_from_caller):
        """Mock function_setup that preserves original kwargs and adds logging."""
        return mock_logging_obj, kwargs_from_caller

    with patch(
        "litellm.a2a_protocol.main.create_a2a_client",
        side_effect=mock_create_a2a_client,
    ), patch(
        "litellm.utils.function_setup",
        side_effect=mock_function_setup,
    ), patch(
        "litellm.a2a_protocol.main.LiteLLMSendMessageResponse.from_a2a_response",
        return_value=MagicMock(),
    ):
        await asend_message(
            request=request,
            api_base="http://localhost:10001",
            metadata={"trace_id": trace_id},
        )

    assert captured_headers.get("X-LiteLLM-Trace-Id") == trace_id, (
        f"Expected X-LiteLLM-Trace-Id={trace_id!r}, "
        f"got {captured_headers.get('X-LiteLLM-Trace-Id')!r}. "
        f"asend_message generated a new UUID instead of reusing the caller's trace_id."
    )
