"""
Tests for PR 4: fix/presidio-anthropic-sse-unmask

Covers:
  - Patch 6a/6b: AnthropicMessagesHandler.process_output_response() accepts
    request_data and merges it so pii_tokens are preserved.
  - Patch 3: async_post_call_streaming_iterator_hook() detects Anthropic native
    SSE bytes chunks and unmasks PII tokens in them.
"""

import json
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_guardrail(output_parse_pii: bool = True):
    """Return a minimal _OPTIONAL_PresidioPIIMasking instance (no live Presidio)."""
    from litellm.proxy.guardrails.guardrail_hooks.presidio import (
        _OPTIONAL_PresidioPIIMasking,
    )

    guardrail = _OPTIONAL_PresidioPIIMasking.__new__(_OPTIONAL_PresidioPIIMasking)
    guardrail.output_parse_pii = output_parse_pii
    guardrail.apply_to_output = False
    guardrail.presidio_analyzer_api_base = "http://fake"
    guardrail.presidio_anonymizer_api_base = "http://fake"
    guardrail.mock_testing = False
    return guardrail


def _make_sse_bytes(text: str, index: int = 0) -> bytes:
    """Build a minimal Anthropic SSE bytes payload containing one text_delta event."""
    delta_payload = {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": text},
    }
    delta_event = (
        b"event: content_block_delta\n"
        b"data: " + json.dumps(delta_payload).encode() + b"\n"
    )
    stop_event = b"event: message_stop\ndata: " + json.dumps({"type": "message_stop"}).encode() + b"\n"
    return delta_event + b"\n\n" + stop_event + b"\n\n"


# ---------------------------------------------------------------------------
# Test 1: process_output_response merges caller request_data (Patch 6a/6b)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_handler_merges_request_data_with_pii_tokens():
    """
    AnthropicMessagesHandler.process_output_response() should preserve pii_tokens
    from the caller-supplied request_data so that apply_guardrail() can unmask them.
    """
    from litellm.llms.anthropic.chat.guardrail_translation.handler import (
        AnthropicMessagesHandler,
    )

    handler = AnthropicMessagesHandler()

    # Simulate the token map built during input masking
    pii_token = "<PERSON>_abc123456789"
    pii_tokens = {pii_token: "Jane Doe"}

    # A fake guardrail that records what request_data it received
    received_request_data = {}

    async def fake_apply_guardrail(inputs, request_data, input_type, logging_obj=None):
        received_request_data.update(request_data or {})
        # Return texts unchanged so the handler can still run its text-update logic
        return dict(inputs)

    mock_guardrail = MagicMock()
    mock_guardrail.apply_guardrail = AsyncMock(side_effect=fake_apply_guardrail)

    # Build a minimal dict-style Anthropic response with a text content block
    response = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20241022",
        "content": [{"type": "text", "text": f"Hello {pii_token}"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 10},
    }

    # Pass caller request_data that includes pii_tokens
    caller_request_data = {"pii_tokens": pii_tokens, "messages": []}

    await handler.process_output_response(
        response=response,
        guardrail_to_apply=mock_guardrail,
        request_data=caller_request_data,
    )

    assert "pii_tokens" in received_request_data, (
        "pii_tokens should be forwarded from caller request_data into apply_guardrail()"
    )
    assert received_request_data["pii_tokens"] == pii_tokens
    # "response" key must also be set (local requirement)
    assert "response" in received_request_data


# ---------------------------------------------------------------------------
# Test 2: streaming hook handles bytes chunks and unmasks PII (Patch 3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_hook_unmasks_anthropic_sse_bytes():
    """
    async_post_call_streaming_iterator_hook() should detect Anthropic native SSE
    bytes chunks, extract text_delta text, unmask pii_tokens, and yield a rebuilt
    SSE bytes chunk with the original text restored.
    """
    guardrail = _make_guardrail(output_parse_pii=True)

    pii_token = "<PERSON>_deadbeef0000"
    original_name = "Alice Smith"
    pii_tokens = {pii_token: original_name}

    # Build SSE bytes with the PII token embedded in the text
    masked_text = f"Hello, {pii_token}! Nice to meet you."
    sse_bytes = _make_sse_bytes(masked_text)

    # The response async generator yields the bytes chunk
    async def fake_response():
        yield sse_bytes

    request_data = {"pii_tokens": pii_tokens}

    from litellm.proxy._types import UserAPIKeyAuth

    user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)

    chunks = []
    async for chunk in guardrail.async_post_call_streaming_iterator_hook(
        user_api_key_dict=user_api_key_dict,
        response=fake_response(),
        request_data=request_data,
    ):
        chunks.append(chunk)

    # The hook yields one chunk per SSE event: the merged text_delta event and
    # the original message_stop event, so 2 chunks total.
    assert len(chunks) == 2, f"Expected 2 rebuilt SSE chunks, got {len(chunks)}"
    text_delta_chunk = chunks[0]
    assert isinstance(text_delta_chunk, bytes), "Output should be bytes for Anthropic SSE path"

    # The rebuilt text_delta chunk must contain the original name, not the PII token
    result_text = text_delta_chunk.decode()
    assert original_name in result_text, (
        f"Expected original name '{original_name}' in rebuilt SSE, got: {result_text}"
    )
    assert pii_token not in result_text, (
        f"PII token '{pii_token}' should be replaced in rebuilt SSE"
    )


# ---------------------------------------------------------------------------
# Test 3: streaming hook pass-through when output_parse_pii is False (Patch 3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_hook_passthrough_when_output_parse_pii_disabled():
    """
    When output_parse_pii=False, bytes chunks should pass through unmodified.
    """
    guardrail = _make_guardrail(output_parse_pii=False)

    pii_token = "<PERSON>_deadbeef0000"
    pii_tokens = {pii_token: "Alice Smith"}
    sse_bytes = _make_sse_bytes(f"Hello, {pii_token}!")

    async def fake_response():
        yield sse_bytes

    request_data = {"pii_tokens": pii_tokens}

    from litellm.proxy._types import UserAPIKeyAuth

    user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)

    chunks = []
    async for chunk in guardrail.async_post_call_streaming_iterator_hook(
        user_api_key_dict=user_api_key_dict,
        response=fake_response(),
        request_data=request_data,
    ):
        chunks.append(chunk)

    # output_parse_pii=False → no processing, chunks pass through
    assert chunks == [sse_bytes], (
        "SSE bytes should pass through unchanged when output_parse_pii is disabled"
    )


# ---------------------------------------------------------------------------
# Test 4: streaming hook with multi-chunk token spanning (Patch 3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_hook_unmasks_token_split_across_chunks():
    """
    PII token strings can be split across multiple SSE delta events.
    _unmask_pii_text() is called on the concatenated text so the token is
    correctly restored even when split.
    """
    guardrail = _make_guardrail(output_parse_pii=True)

    pii_token = "<PERSON>_splittoken01"
    original_name = "Bob Jones"
    pii_tokens = {pii_token: original_name}

    # Split the token across two SSE chunks
    half = len(pii_token) // 2
    first_half = pii_token[:half]
    second_half = pii_token[half:]

    def _raw_delta(text: str, index: int = 0) -> bytes:
        payload = {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "text_delta", "text": text},
        }
        return b"event: content_block_delta\ndata: " + json.dumps(payload).encode() + b"\n"

    chunk1 = _raw_delta(f"Hello, {first_half}") + b"\n\n"
    chunk2 = _raw_delta(second_half) + b"\n\n"

    async def fake_response():
        yield chunk1
        yield chunk2

    request_data = {"pii_tokens": pii_tokens}

    from litellm.proxy._types import UserAPIKeyAuth

    user_api_key_dict = MagicMock(spec=UserAPIKeyAuth)

    chunks = []
    async for chunk in guardrail.async_post_call_streaming_iterator_hook(
        user_api_key_dict=user_api_key_dict,
        response=fake_response(),
        request_data=request_data,
    ):
        chunks.append(chunk)

    assert len(chunks) == 1
    result_text = chunks[0].decode()
    assert original_name in result_text, (
        f"Split token should be reassembled and unmasked. Got: {result_text}"
    )
    assert pii_token not in result_text
