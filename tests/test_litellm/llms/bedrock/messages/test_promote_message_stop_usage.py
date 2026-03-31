"""
Tests for _promote_message_stop_usage in Bedrock Anthropic Messages streaming.

Bedrock reports cache usage fields (cache_creation_input_tokens,
cache_read_input_tokens) only on message_stop. Claude Code's SDK only
reads usage from message_start and message_delta. This test verifies
that the promotion logic correctly merges those fields into message_delta
before yielding.
"""

import pytest

from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaudeMessagesConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _collect(async_iter):
    results = []
    async for item in async_iter:
        results.append(item)
    return results


async def _async_iter_from_list(items):
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# Tests for _promote_message_stop_usage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_fields_promoted_from_message_stop_to_message_delta():
    """
    Simulates the Bedrock streaming pattern where message_stop carries
    cache_creation_input_tokens and cache_read_input_tokens but
    message_delta only has output_tokens.
    """
    chunks = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hello!"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 12},
        },
        {
            "type": "message_stop",
            "usage": {
                "input_tokens": 3,
                "cache_creation_input_tokens": 30457,
                "cache_read_input_tokens": 0,
                "output_tokens": 12,
            },
        },
    ]

    patched = await _collect(
        AmazonAnthropicClaudeMessagesConfig._promote_message_stop_usage(
            _async_iter_from_list(chunks)
        )
    )

    # Find the message_delta in the output
    delta_chunk = next(
        c for c in patched if isinstance(c, dict) and c.get("type") == "message_delta"
    )

    assert delta_chunk["usage"]["output_tokens"] == 12
    assert delta_chunk["usage"]["input_tokens"] == 3 + 30457 + 0
    assert delta_chunk["usage"]["cache_creation_input_tokens"] == 30457
    assert delta_chunk["usage"]["cache_read_input_tokens"] == 0


@pytest.mark.asyncio
async def test_cache_read_promoted_on_warm_cache():
    """
    When cache is warm, cache_read_input_tokens is large and
    cache_creation_input_tokens is small. Both should be promoted.
    """
    chunks = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_test2",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 0},
            },
        },
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 302},
        },
        {
            "type": "message_stop",
            "usage": {
                "input_tokens": 3,
                "cache_creation_input_tokens": 23,
                "cache_read_input_tokens": 29396,
                "output_tokens": 302,
            },
        },
    ]

    patched = await _collect(
        AmazonAnthropicClaudeMessagesConfig._promote_message_stop_usage(
            _async_iter_from_list(chunks)
        )
    )

    delta_chunk = next(
        c for c in patched if isinstance(c, dict) and c.get("type") == "message_delta"
    )

    assert delta_chunk["usage"]["cache_creation_input_tokens"] == 23
    assert delta_chunk["usage"]["cache_read_input_tokens"] == 29396
    assert delta_chunk["usage"]["input_tokens"] == 3 + 23 + 29396
    assert delta_chunk["usage"]["output_tokens"] == 302


@pytest.mark.asyncio
async def test_no_cache_fields_passes_through_unchanged():
    """
    When message_stop has no cache fields (e.g. non-Bedrock or no caching),
    message_delta should pass through unchanged.
    """
    chunks = [
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 50},
        },
        {
            "type": "message_stop",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        },
    ]

    patched = await _collect(
        AmazonAnthropicClaudeMessagesConfig._promote_message_stop_usage(
            _async_iter_from_list(chunks)
        )
    )

    delta_chunk = next(
        c for c in patched if isinstance(c, dict) and c.get("type") == "message_delta"
    )

    assert delta_chunk["usage"]["output_tokens"] == 50
    assert delta_chunk["usage"]["input_tokens"] == 100
    assert "cache_creation_input_tokens" not in delta_chunk["usage"]
    assert "cache_read_input_tokens" not in delta_chunk["usage"]


@pytest.mark.asyncio
async def test_input_tokens_includes_cache_totals():
    """
    Bedrock reports only uncached tokens in input_tokens. After promotion,
    input_tokens must equal uncached + cache_creation + cache_read so
    clients see the true total (matching Anthropic direct API convention).
    """
    chunks = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_total",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 0},
            },
        },
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {
                "output_tokens": 231,
                "input_tokens": 3,
                "cache_creation_input_tokens": 1745,
                "cache_read_input_tokens": 30456,
            },
        },
        {
            "type": "message_stop",
            "usage": {
                "input_tokens": 3,
                "cache_creation_input_tokens": 1745,
                "cache_read_input_tokens": 30456,
                "output_tokens": 231,
            },
        },
    ]

    patched = await _collect(
        AmazonAnthropicClaudeMessagesConfig._promote_message_stop_usage(
            _async_iter_from_list(chunks)
        )
    )

    delta_chunk = next(
        c for c in patched if isinstance(c, dict) and c.get("type") == "message_delta"
    )

    assert delta_chunk["usage"]["input_tokens"] == 3 + 1745 + 30456
    assert delta_chunk["usage"]["cache_creation_input_tokens"] == 1745
    assert delta_chunk["usage"]["cache_read_input_tokens"] == 30456
    assert delta_chunk["usage"]["output_tokens"] == 231


@pytest.mark.asyncio
async def test_chunk_order_preserved():
    """All chunks should be yielded in order."""
    chunks = [
        {
            "type": "message_start",
            "message": {"usage": {"input_tokens": 3, "output_tokens": 0}},
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hi"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 5},
        },
        {
            "type": "message_stop",
            "usage": {
                "input_tokens": 3,
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 0,
                "output_tokens": 5,
            },
        },
    ]

    patched = await _collect(
        AmazonAnthropicClaudeMessagesConfig._promote_message_stop_usage(
            _async_iter_from_list(chunks)
        )
    )

    types = [c.get("type") if isinstance(c, dict) else None for c in patched]
    assert types == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
