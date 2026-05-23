"""
Tests for AnthropicResponsesStreamWrapper reasoning -> signature_delta emission.

Covers Phase 3.4 of the encrypted-reasoning round-trip fix: when the upstream
Responses API streams a reasoning item with `encrypted_content`, the wrapper
must emit a `signature_delta` carrying the packed (id + encrypted_content)
blob BEFORE the reasoning block's `content_block_stop`. Without this, the
client (e.g. Claude Code) has no way to replay the reasoning item on
subsequent turns, breaking reasoning continuity for the openai/chatgpt
providers.
"""

import asyncio

from litellm.llms.anthropic.experimental_pass_through.responses_adapters.streaming_iterator import (
    AnthropicResponsesStreamWrapper,
)
from litellm.llms.anthropic.experimental_pass_through.responses_adapters.transformation import (
    _pack_signature,
    _unpack_signature,
)


class _AsyncEventStream:
    """Minimal async iterator over a static list of event dicts."""

    def __init__(self, events):
        self._events = list(events)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)


def _collect(wrapper):
    async def _drive():
        out = []
        async for chunk in wrapper:
            out.append(chunk)
        return out

    return asyncio.run(_drive())


def _build_stream_with_reasoning(rs_id: str, enc: str):
    """Construct a synthetic Responses SSE event sequence with a reasoning
    item carrying encrypted_content, followed by a short message item."""
    return [
        {"type": "response.created"},
        {
            "type": "response.output_item.added",
            "item": {"id": rs_id, "type": "reasoning"},
        },
        {
            "type": "response.reasoning_summary_text.delta",
            "item_id": rs_id,
            "delta": "thinking about it",
        },
        {
            "type": "response.output_item.done",
            "item": {
                "id": rs_id,
                "type": "reasoning",
                "encrypted_content": enc,
                "summary": [{"type": "summary_text", "text": "thinking about it"}],
            },
        },
        {
            "type": "response.output_item.added",
            "item": {"id": "msg_1", "type": "message"},
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "delta": "hi",
        },
        {
            "type": "response.output_item.done",
            "item": {"id": "msg_1", "type": "message"},
        },
        {
            "type": "response.completed",
            "response": {"status": "completed", "output": []},
        },
    ]


def test_reasoning_emits_signature_delta_before_content_block_stop():
    rs_id = "rs_test_abc"
    enc = "ENCRYPTED-PAYLOAD-XYZ"
    events = _build_stream_with_reasoning(rs_id, enc)
    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_AsyncEventStream(events), model="gpt-5.5"
    )

    chunks = _collect(wrapper)

    # Locate the reasoning block start
    start_idx = next(
        i
        for i, c in enumerate(chunks)
        if c.get("type") == "content_block_start"
        and c["content_block"]["type"] == "thinking"
    )
    block_index = chunks[start_idx]["index"]

    # The first content_block_stop AFTER the thinking start belongs to the
    # reasoning block. Immediately before it, we must see signature_delta.
    stop_positions = [
        i
        for i, c in enumerate(chunks[start_idx:], start=start_idx)
        if c.get("type") == "content_block_stop" and c.get("index") == block_index
    ]
    assert stop_positions, "missing content_block_stop for reasoning block"
    stop_idx = stop_positions[0]

    sig_event = chunks[stop_idx - 1]
    assert sig_event["type"] == "content_block_delta"
    assert sig_event["index"] == block_index
    assert sig_event["delta"]["type"] == "signature_delta"

    # Signature must round-trip via the codec back to the originating id+enc.
    unpacked_id, unpacked_enc = _unpack_signature(sig_event["delta"]["signature"])
    assert unpacked_id == rs_id
    assert unpacked_enc == enc
    assert sig_event["delta"]["signature"] == _pack_signature(rs_id, enc)


def test_reasoning_without_encrypted_content_skips_signature_delta():
    """If encrypted_content is absent, we must NOT emit an empty signature_delta —
    just close the block cleanly. Preserves backward compat for providers
    that don't return encrypted reasoning."""
    rs_id = "rs_plain"
    events = [
        {"type": "response.created"},
        {
            "type": "response.output_item.added",
            "item": {"id": rs_id, "type": "reasoning"},
        },
        {
            "type": "response.output_item.done",
            "item": {"id": rs_id, "type": "reasoning"},  # no encrypted_content
        },
        {
            "type": "response.completed",
            "response": {"status": "completed", "output": []},
        },
    ]
    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_AsyncEventStream(events), model="gpt-5.5"
    )

    chunks = _collect(wrapper)

    # No signature_delta anywhere in the stream.
    assert not any(
        c.get("type") == "content_block_delta"
        and c.get("delta", {}).get("type") == "signature_delta"
        for c in chunks
    )
    # But the content_block_stop for the reasoning block is still emitted.
    assert any(c.get("type") == "content_block_stop" for c in chunks)


def test_message_only_stream_unaffected():
    """A response with no reasoning item must not emit any signature_delta."""
    events = [
        {"type": "response.created"},
        {
            "type": "response.output_item.added",
            "item": {"id": "msg_1", "type": "message"},
        },
        {
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "delta": "hello",
        },
        {
            "type": "response.output_item.done",
            "item": {"id": "msg_1", "type": "message"},
        },
        {
            "type": "response.completed",
            "response": {"status": "completed", "output": []},
        },
    ]
    wrapper = AnthropicResponsesStreamWrapper(
        responses_stream=_AsyncEventStream(events), model="gpt-5.5"
    )

    chunks = _collect(wrapper)

    assert not any(
        c.get("type") == "content_block_delta"
        and c.get("delta", {}).get("type") == "signature_delta"
        for c in chunks
    )
