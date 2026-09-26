"""
Regression test for https://github.com/BerriAI/litellm/issues/43010

/v1/responses streaming with Anthropic doubles the thinking text that ends up
in the reasoning item's encrypted_content.

Root cause: the Anthropic stream handler emits one thinking_blocks entry per
`thinking_delta` (carrying that delta's text), and then on the `signature_delta`
emits a *second* thinking_blocks entry whose `thinking` field repeats the FULL
concatenated text of the block (handler.py ~ lines 701-716). The stream chunk
builder (`ChunkProcessor._build_thinking_blocks`) unconditionally appends every
entry's `thinking` text before flushing on the signature, so the final block's
text = (sum of deltas == full text) + (full text from the signature entry)
== the text twice.

These tests drive the REAL handler chunk_parser on raw Anthropic SSE events and
the REAL ChunkProcessor. No API key / network needed.
"""

from litellm.litellm_core_utils.streaming_chunk_builder_utils import ChunkProcessor
from litellm.llms.anthropic.chat.handler import ModelResponseIterator


THINKING_DELTAS = [
    "The user is asking ",
    "about the weather. ",
    "I should reason step by step.",
]
FULL_THINKING = "".join(THINKING_DELTAS)
SIGNATURE = "ErUBCkYIBRgCKkD_signature_blob_xyz"


def _drive_handler_on_anthropic_thinking_stream():
    """Feed raw Anthropic thinking-stream SSE events through the real handler,
    returning the ModelResponseStream chunks it produces (mirrors production)."""
    iterator = ModelResponseIterator(streaming_response=None, sync_stream=True)

    raw_events = [
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
    ]
    for delta in THINKING_DELTAS:
        raw_events.append(
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": delta}}
        )
    # signature_delta: handler joins all accumulated thinking blocks into the
    # signed block's `thinking` -> the full text again.
    raw_events.append(
        {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": SIGNATURE}}
    )

    chunks = []
    for ev in raw_events:
        parsed = iterator.chunk_parser(ev)
        if parsed is not None:
            chunks.append(parsed)
    return chunks


def test_handler_signature_entry_repeats_full_thinking_text():
    """Documents the handler behavior the builder must tolerate: the signature
    entry carries the full text, so a naive concat doubles it."""
    chunks = _drive_handler_on_anthropic_thinking_stream()
    signed_texts = [
        tb.get("thinking")
        for c in chunks
        for tb in (c["choices"][0]["delta"].get("thinking_blocks") or [])
        if tb.get("signature")
    ]
    assert signed_texts == [FULL_THINKING], signed_texts


def test_responses_reasoning_thinking_text_is_not_doubled():
    """The bug: reassembled thinking text must equal the model's text ONCE."""
    chunks = _drive_handler_on_anthropic_thinking_stream()
    thinking_chunks = [c for c in chunks if c["choices"][0]["delta"].get("thinking_blocks")]
    result = ChunkProcessor(chunks=chunks).get_combined_thinking_content(thinking_chunks)

    assert result is not None and len(result) == 1, result
    block = result[0]
    assert block["type"] == "thinking"
    assert block["signature"] == SIGNATURE
    # FAILS on current main: got == FULL_THINKING + FULL_THINKING
    assert block["thinking"] == FULL_THINKING, (
        f"thinking text doubled: expected {FULL_THINKING!r}, got {block['thinking']!r}"
    )
    assert block["thinking"].count(FULL_THINKING) == 1


def _thinking_chunk(blocks):
    """A ModelResponseStream-shaped chunk carrying thinking_blocks (builder input)."""
    return {"choices": [{"delta": {"thinking_blocks": blocks}}]}


def test_signature_only_stream_text_preserved():
    """Fallback: when the full text arrives ONLY on the signature entry (no separate
    deltas), it is authoritative and must be preserved, not dropped."""
    chunks = [_thinking_chunk([{"type": "thinking", "thinking": FULL_THINKING, "signature": SIGNATURE}])]
    result = ChunkProcessor(chunks=chunks).get_combined_thinking_content(chunks)
    assert result is not None and len(result) == 1, result
    assert result[0]["thinking"] == FULL_THINKING
    assert result[0]["signature"] == SIGNATURE


def test_bedrock_style_empty_signature_entry_does_not_truncate():
    """Other-provider shape: Bedrock emits the signature entry with thinking="" while
    the real text arrived as deltas. The block text must survive intact (no truncation,
    no doubling) -- guarding the fix against the truncation failure mode."""
    chunks = [
        _thinking_chunk([{"type": "thinking", "thinking": "The user is asking "}]),
        _thinking_chunk([{"type": "thinking", "thinking": "about the weather."}]),
        _thinking_chunk([{"type": "thinking", "thinking": "", "signature": SIGNATURE}]),
    ]
    result = ChunkProcessor(chunks=chunks).get_combined_thinking_content(chunks)
    assert result is not None and len(result) == 1, result
    assert result[0]["thinking"] == "The user is asking about the weather."
    assert result[0]["signature"] == SIGNATURE
