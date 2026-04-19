"""
Tests for spend log creation when using the native /v1/messages handler.

Covers the fixes in the cherry-picked commit:
1. async_success_handler builds StandardLoggingPayload for call_type="anthropic_messages"
   (not just "pass_through_endpoint").
2. response_cost from kwargs is preserved when already set by the pass-through handler.
3. _build_standard_logging_payload overlays anthropic_raw_response on SLP["response"].
4. _extract_response_id_from_chunks extracts msg_* ID from SSE lines.
5. JSONDecodeError on "data: [DONE]" lines does not abort chunk iteration.
"""

import json
import os
import sys
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler import (
    AnthropicPassthroughLoggingHandler,
)
from litellm.types.utils import CallTypes, ModelResponse, Usage


# ---------------------------------------------------------------------------
# 1. async_success_handler builds SLP for anthropic_messages (non-streaming)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_success_handler_builds_slp_for_anthropic_messages():
    """
    async_success_handler should build standard_logging_object for
    call_type='anthropic_messages' (non-streaming), just as it does for
    'pass_through_endpoint'. Before the fix, the elif branch only matched
    'pass_through_endpoint'.
    """
    logging_obj = LiteLLMLoggingObj(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "Hi"}],
        stream=False,
        call_type=CallTypes.anthropic_messages.value,
        start_time=datetime.now(),
        litellm_call_id="test-slp-anthropic",
        function_id="test-fn-slp",
    )

    logging_obj.model_call_details = {
        "litellm_params": {"metadata": {}, "proxy_server_request": {}},
        "litellm_call_id": "test-slp-anthropic",
        "response_cost": 0.0042,
        "custom_llm_provider": "anthropic",
    }

    result = ModelResponse(
        id="msg_test_slp",
        model="claude-sonnet-4-20250514",
        usage=Usage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
    )

    start_time = datetime.now()
    end_time = datetime.now()

    with patch.object(logging_obj, "get_combined_callback_list", return_value=[]):
        await logging_obj.async_success_handler(
            result=result,
            start_time=start_time,
            end_time=end_time,
            cache_hit=False,
        )

    assert "standard_logging_object" in logging_obj.model_call_details, (
        "standard_logging_object must be set for call_type='anthropic_messages'"
    )
    assert logging_obj.model_call_details["standard_logging_object"] is not None


# ---------------------------------------------------------------------------
# 2. response_cost from kwargs is preserved for anthropic_messages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_success_handler_preserves_response_cost_for_anthropic_messages():
    """
    When the pass-through handler pre-calculates response_cost and stores it
    in model_call_details, the anthropic_messages branch must not overwrite
    it to None. Mirrors the existing test for 'pass_through_endpoint'.
    """
    logging_obj = LiteLLMLoggingObj(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "test"}],
        stream=False,
        call_type=CallTypes.anthropic_messages.value,
        start_time=datetime.now(),
        litellm_call_id="test-cost-preserve",
        function_id="test-fn-cost",
    )

    logging_obj.model_call_details = {
        "litellm_params": {"metadata": {}, "proxy_server_request": {}},
        "litellm_call_id": "test-cost-preserve",
        "response_cost": 0.0035,
        "custom_llm_provider": "anthropic",
    }

    result = ModelResponse(
        id="msg_test_cost",
        model="claude-sonnet-4-20250514",
        usage=Usage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
    )

    start_time = datetime.now()
    end_time = datetime.now()

    with patch.object(logging_obj, "get_combined_callback_list", return_value=[]):
        await logging_obj.async_success_handler(
            result=result,
            start_time=start_time,
            end_time=end_time,
            cache_hit=False,
        )

    # response_cost must be preserved, not overwritten to None
    assert logging_obj.model_call_details.get("response_cost") is not None
    assert logging_obj.model_call_details["response_cost"] > 0

    # standard_logging_object should also have the cost
    slo = logging_obj.model_call_details.get("standard_logging_object")
    assert slo is not None
    assert slo["response_cost"] > 0


# ---------------------------------------------------------------------------
# 3. _build_standard_logging_payload overlays anthropic_raw_response
# ---------------------------------------------------------------------------


def test_build_slp_overlays_anthropic_raw_response():
    """
    When model_call_details contains 'anthropic_raw_response',
    _build_standard_logging_payload should overwrite payload['response']
    with the raw Anthropic JSON dict.
    """
    logging_obj = LiteLLMLoggingObj(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "Hi"}],
        stream=False,
        call_type=CallTypes.anthropic_messages.value,
        start_time=time.time(),
        litellm_call_id="test-raw-overlay",
        function_id="test-fn-overlay",
    )

    raw_anthropic = {
        "id": "msg_01XYZ",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Hello!"}],
        "model": "claude-sonnet-4-20250514",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    logging_obj.model_call_details["anthropic_raw_response"] = raw_anthropic

    result = ModelResponse(
        id="chatcmpl-abc",
        model="claude-sonnet-4-20250514",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    payload = logging_obj._build_standard_logging_payload(
        init_response_obj=result,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    assert payload is not None
    assert payload["response"] == raw_anthropic, (
        "SLP response should be the raw Anthropic JSON, not the chat.completion shape"
    )


def test_build_slp_no_overlay_without_anthropic_raw():
    """
    When model_call_details does NOT contain 'anthropic_raw_response',
    the default SLP response should be used (no overlay).
    """
    logging_obj = LiteLLMLoggingObj(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": "Hi"}],
        stream=False,
        call_type=CallTypes.anthropic_messages.value,
        start_time=time.time(),
        litellm_call_id="test-no-overlay",
        function_id="test-fn-no-overlay",
    )

    result = ModelResponse(
        id="chatcmpl-abc",
        model="claude-sonnet-4-20250514",
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    payload = logging_obj._build_standard_logging_payload(
        init_response_obj=result,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    assert payload is not None
    # response should NOT be a dict with "type": "message" (Anthropic shape)
    if isinstance(payload["response"], dict):
        assert payload["response"].get("type") != "message"


# ---------------------------------------------------------------------------
# 4. _extract_response_id_from_chunks
# ---------------------------------------------------------------------------


class TestExtractResponseIdFromChunks:
    """Tests for AnthropicPassthroughLoggingHandler._extract_response_id_from_chunks."""

    def test_extracts_id_from_message_start_event(self):
        chunks = [
            'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_01ABC","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}',
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',
            'event: message_stop\ndata: {"type":"message_stop"}',
        ]
        result = AnthropicPassthroughLoggingHandler._extract_response_id_from_chunks(
            chunks
        )
        assert result == "msg_01ABC"

    def test_returns_none_for_no_message_start(self):
        chunks = [
            'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',
            'event: message_stop\ndata: {"type":"message_stop"}',
        ]
        result = AnthropicPassthroughLoggingHandler._extract_response_id_from_chunks(
            chunks
        )
        assert result is None

    def test_handles_bytes_chunks(self):
        chunks = [
            b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_02DEF","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[],"usage":{"input_tokens":5,"output_tokens":0}}}',
        ]
        result = AnthropicPassthroughLoggingHandler._extract_response_id_from_chunks(
            chunks
        )
        assert result == "msg_02DEF"

    def test_returns_none_for_empty_chunks(self):
        result = AnthropicPassthroughLoggingHandler._extract_response_id_from_chunks([])
        assert result is None

    def test_returns_none_for_malformed_json(self):
        chunks = [
            'event: message_start\ndata: {not valid json}',
        ]
        result = AnthropicPassthroughLoggingHandler._extract_response_id_from_chunks(
            chunks
        )
        assert result is None


# ---------------------------------------------------------------------------
# 5. JSONDecodeError on "data: [DONE]" does not abort chunk iteration
# ---------------------------------------------------------------------------


def test_split_sse_done_line_does_not_raise():
    """
    GitHub Copilot appends 'data: [DONE]' (OpenAI format) at the end of
    Anthropic SSE streams. The chunk parser must skip it gracefully
    instead of raising json.JSONDecodeError.
    """
    chunks_with_done = [
        'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_01GHI","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}',
        'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}',
        'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}',
        'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}',
        'event: message_stop\ndata: {"type":"message_stop"}',
        "data: [DONE]",
    ]

    # _split_sse_chunk_into_events handles individual lines;
    # the JSONDecodeError fix is in _handle_logging_anthropic_collected_chunks.
    # We just verify the static helper doesn't crash on the [DONE] line.
    for chunk in chunks_with_done:
        events = AnthropicPassthroughLoggingHandler._split_sse_chunk_into_events(chunk)
        # Should not raise; events is a list of strings


def test_anthropic_passthrough_handler_processes_chunks_with_done_line():
    """
    End-to-end: _handle_logging_anthropic_collected_chunks should produce
    a valid result dict even when chunks include 'data: [DONE]'.
    """
    chunks = [
        'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_01JKL","type":"message","role":"assistant","model":"claude-sonnet-4-20250514","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}',
        'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
        'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello world"}}',
        'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}',
        'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":5}}',
        'event: message_stop\ndata: {"type":"message_stop"}',
        "data: [DONE]",
    ]

    mock_logging_obj = MagicMock()
    mock_logging_obj.model_call_details = {"model": "claude-sonnet-4-20250514"}
    mock_logging_obj.litellm_call_id = "test-done-line"

    mock_passthrough_handler = MagicMock()

    result = AnthropicPassthroughLoggingHandler._handle_logging_anthropic_collected_chunks(
        litellm_logging_obj=mock_logging_obj,
        passthrough_success_handler_obj=mock_passthrough_handler,
        url_route="/v1/messages",
        request_body={"model": "claude-sonnet-4-20250514"},
        endpoint_type="messages",
        start_time=datetime.now(),
        all_chunks=chunks,
        end_time=datetime.now(),
    )

    assert result is not None
    assert "result" in result
    assert "kwargs" in result
