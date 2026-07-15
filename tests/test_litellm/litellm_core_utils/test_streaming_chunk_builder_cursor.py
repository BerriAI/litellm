"""
Regression tests for the Anthropic message_start cursor=1 bug in
ChunkProcessor._calculate_usage_per_chunk.

Background
----------
Anthropic streams a `message_start` event that carries
`usage.output_tokens=1` as a placeholder ("cursor"). The real cumulative
output count only arrives in the final `message_delta` event. When a
stream is cancelled before `message_delta` lands (very common for
thinking models on long-tail prompts), the last-wins accumulator in
ChunkProcessor leaves completion_tokens stuck at 1. Because 1 is
truthy, the `completion_tokens or token_counter(text=...)` fallback in
calculate_usage() never fires, and the request is billed for 1 output
token even when several thousand tokens of text were actually streamed.

These tests pin the post-fix behavior: completion_tokens should reset
to 0 when the only update we saw was the cursor, allowing the
text-based fallback to estimate from the real completion text.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.litellm_core_utils.streaming_chunk_builder_utils import ChunkProcessor
from litellm.types.utils import (
    Delta,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)


def _make_chunk(
    *,
    content: str = "",
    usage: Usage = None,
    finish_reason: str = None,
    custom_llm_provider: str = "anthropic",
) -> ModelResponseStream:
    chunk = ModelResponseStream(
        id="msg_test",
        created=1738900000,
        model="claude-sonnet-4-6",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=finish_reason,
                index=0,
                delta=Delta(content=content, role="assistant"),
            )
        ],
        usage=usage,
    )
    # The cursor reset is now gated on provider; populate the same field the
    # real streaming_handler sets (see litellm/litellm_core_utils/streaming_handler.py).
    chunk._hidden_params = {"custom_llm_provider": custom_llm_provider}
    return chunk


class TestAnthropicCursorBug:
    """The core regression: completion_tokens=1 cursor must not leak through."""

    def test_only_message_start_cursor_resets_completion_to_zero(self):
        """
        Stream cancelled before message_delta — only the message_start cursor
        (output_tokens=1) was seen. Per-chunk accumulator must reset to 0 so
        token_counter fallback can estimate from completion text.
        """
        # Anthropic message_start: input_tokens accurate, output_tokens=1 cursor
        message_start = _make_chunk(
            usage=Usage(prompt_tokens=1024, completion_tokens=1, total_tokens=1025)
        )
        # Several content_block_delta chunks (no usage attached)
        text_chunks = [
            _make_chunk(content="Hello"),
            _make_chunk(content=" world"),
            _make_chunk(content=" this is partial."),
        ]
        chunks = [message_start, *text_chunks]

        processor = ChunkProcessor(chunks=chunks, messages=[])
        result = processor._calculate_usage_per_chunk(chunks=chunks)

        assert result["prompt_tokens"] == 1024
        # The cursor value of 1 must NOT leak through — should be reset to 0
        # so the text-based fallback estimates the real completion length.
        assert result["completion_tokens"] == 0, (
            "completion_tokens=1 from message_start cursor leaked through. "
            "Should reset to 0 when only cursor was seen, so token_counter "
            "fallback in calculate_usage() can estimate from completion text."
        )

    def test_message_start_plus_message_delta_uses_delta_value(self):
        """
        Normal complete stream: message_start cursor=1, then message_delta=3847.
        Last-wins must give 3847 (the real value).
        """
        message_start = _make_chunk(
            usage=Usage(prompt_tokens=1024, completion_tokens=1, total_tokens=1025)
        )
        text_chunks = [_make_chunk(content=t) for t in ["Hello", " world", "!"]]
        # message_delta with the real cumulative output_tokens
        message_delta = _make_chunk(
            usage=Usage(prompt_tokens=1024, completion_tokens=3847, total_tokens=4871),
            finish_reason="stop",
        )
        chunks = [message_start, *text_chunks, message_delta]

        processor = ChunkProcessor(chunks=chunks, messages=[])
        result = processor._calculate_usage_per_chunk(chunks=chunks)

        assert result["prompt_tokens"] == 1024
        assert result["completion_tokens"] == 3847

    def test_calculate_usage_falls_back_to_token_counter_for_cursor_only(self):
        """
        End-to-end via calculate_usage(): cursor-only stream + real completion
        text should produce a token-counter estimate, NOT 1.
        """
        message_start = _make_chunk(
            usage=Usage(prompt_tokens=1024, completion_tokens=1, total_tokens=1025)
        )
        # ~50 visible chars ≈ ~12 tokens (anthropic-style tokenizer ballpark)
        text_chunks = [
            _make_chunk(content="Based on your question, I think the answer is "),
            _make_chunk(content="forty-two. Here is my reasoning: "),
        ]
        chunks = [message_start, *text_chunks]
        completion_output = (
            "Based on your question, I think the answer is forty-two. "
            "Here is my reasoning: "
        )

        processor = ChunkProcessor(chunks=chunks, messages=[])
        usage = processor.calculate_usage(
            chunks=chunks,
            model="claude-sonnet-4-6",
            completion_output=completion_output,
            messages=[],
        )

        # Should be a token_counter estimate of the text, not the cursor 1
        assert usage.completion_tokens > 1, (
            f"Expected token_counter estimate of completion text, got "
            f"completion_tokens={usage.completion_tokens} (likely stuck at cursor)"
        )

    def test_cache_fields_preserved_from_message_start(self):
        """cache_read / cache_creation come from message_start and must survive."""
        message_start_usage = Usage(
            prompt_tokens=1024, completion_tokens=1, total_tokens=1025
        )
        # Anthropic puts these in message_start
        message_start_usage.cache_read_input_tokens = 512
        message_start_usage.cache_creation_input_tokens = 128
        message_start = _make_chunk(usage=message_start_usage)

        chunks = [message_start, _make_chunk(content="hi")]
        processor = ChunkProcessor(chunks=chunks, messages=[])
        result = processor._calculate_usage_per_chunk(chunks=chunks)

        assert result["cache_read_input_tokens"] == 512
        assert result["cache_creation_input_tokens"] == 128

    def test_openai_streaming_unaffected(self):
        """
        OpenAI's only usage chunk is the penultimate one (with
        stream_options.include_usage=true), and it carries the real value
        directly. Our cursor fix must not break this path — output > 1
        means saw_non_cursor_completion=True so no reset happens.
        """
        # Simulate OpenAI: content chunks first, then ONE usage chunk at the end
        text_chunks = [_make_chunk(content=t) for t in ["The", " answer", " is 42"]]
        usage_chunk = _make_chunk(
            usage=Usage(prompt_tokens=42, completion_tokens=15, total_tokens=57),
            finish_reason="stop",
        )
        chunks = [*text_chunks, usage_chunk]

        processor = ChunkProcessor(chunks=chunks, messages=[])
        result = processor._calculate_usage_per_chunk(chunks=chunks)

        assert result["prompt_tokens"] == 42
        assert result["completion_tokens"] == 15

    def test_single_token_completion_legitimate_case(self):
        """
        Edge case: a stream that legitimately completes with output_tokens=1
        (e.g., model returns just "Yes."). Without saw_non_cursor_completion
        we'd reset to 0 and fall through to token_counter — but token_counter
        on a 1-token string also gives ~1, so billing is still approximately
        correct. This test pins that the result is sane (1 or 0).
        """
        message_start = _make_chunk(
            usage=Usage(prompt_tokens=20, completion_tokens=1, total_tokens=21)
        )
        text_chunk = _make_chunk(content="Yes.")
        # Anthropic's message_delta also gives output_tokens=1 in this case
        message_delta = _make_chunk(
            usage=Usage(prompt_tokens=20, completion_tokens=1, total_tokens=21),
            finish_reason="stop",
        )
        chunks = [message_start, text_chunk, message_delta]

        processor = ChunkProcessor(chunks=chunks, messages=[])
        usage = processor.calculate_usage(
            chunks=chunks,
            model="claude-sonnet-4-6",
            completion_output="Yes.",
            messages=[],
        )

        # Two completion-bearing usage events (message_start AND message_delta
        # both with output_tokens=1) is positive evidence that message_delta
        # arrived — saw_non_cursor_completion goes True via the count >= 2
        # branch and the reset is suppressed. Result: completion_tokens stays
        # at the legitimate value of 1.
        assert usage.completion_tokens == 1, (
            f"Legitimate single-token completion should bill exactly 1 token "
            f"(message_start + message_delta both saw output_tokens=1, "
            f"confirming message_delta arrived), got {usage.completion_tokens}"
        )

    def test_anthropic_cache_only_chunks_after_message_start_still_resets(self):
        """
        Cache-only chunks (cache_read_input_tokens > 0 but completion_tokens=0)
        following message_start should not be mistaken for completion progress.
        The cursor=1 from message_start stays the only completion update; reset
        must fire so token_counter estimates from completion text instead of
        billing the placeholder.
        """
        message_start_usage = Usage(
            prompt_tokens=1024, completion_tokens=1, total_tokens=1025
        )
        message_start_usage.cache_read_input_tokens = 4096
        message_start = _make_chunk(usage=message_start_usage)
        # Subsequent chunks with cache fields but no completion_tokens
        cache_chunk_usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        cache_chunk_usage.cache_read_input_tokens = 4096
        cache_chunk = _make_chunk(content="partial", usage=cache_chunk_usage)
        # No message_delta — stream was cancelled
        chunks = [message_start, cache_chunk]

        processor = ChunkProcessor(chunks=chunks, messages=[])
        result = processor._calculate_usage_per_chunk(chunks=chunks)

        assert result["cache_read_input_tokens"] == 4096
        assert result["completion_tokens"] == 0, (
            "cache chunks alone don't count as completion progress — only "
            "completion_tokens > 0 in a usage event proves real output happened. "
            "Reset to 0 forces token_counter fallback."
        )


class TestProviderGuard:
    """Class A: the cursor-reset heuristic must NOT silently affect non-Anthropic
    providers, even if they happen to report completion_tokens=1."""

    def test_non_anthropic_provider_completion_tokens_one_not_reset(self):
        """
        Some non-Anthropic provider legitimately reports completion_tokens=1
        in its single usage chunk. Without the provider guard the cursor
        heuristic would silently reset it to 0 and bill via token_counter,
        producing a different (often inflated) number than what the provider
        actually charged.
        """
        chunks = [
            _make_chunk(
                usage=Usage(prompt_tokens=10, completion_tokens=1, total_tokens=11),
                finish_reason="stop",
                custom_llm_provider="openai",
            ),
        ]
        processor = ChunkProcessor(chunks=chunks, messages=[])
        result = processor._calculate_usage_per_chunk(chunks=chunks)
        assert result["completion_tokens"] == 1, (
            "Non-Anthropic providers must not be subject to the message_start "
            "cursor reset — their completion_tokens=1 is the real value."
        )

    def test_unknown_provider_completion_tokens_one_not_reset(self):
        """No custom_llm_provider on hidden_params (older path or custom
        plugin) — heuristic must not fire."""
        chunk = _make_chunk(
            usage=Usage(prompt_tokens=10, completion_tokens=1, total_tokens=11),
        )
        # Explicitly clear hidden_params to simulate the unknown-provider case
        chunk._hidden_params = {}
        processor = ChunkProcessor(chunks=[chunk], messages=[])
        result = processor._calculate_usage_per_chunk(chunks=[chunk])
        assert result["completion_tokens"] == 1


class TestNonAnthropicStreamingIntact:
    """Make sure providers without cursor pattern still work."""

    def test_completion_tokens_above_one_never_resets(self):
        """Any chunk reporting completion_tokens > 1 sets saw_non_cursor
        and prevents the reset."""
        chunks = [
            _make_chunk(
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
            ),
        ]
        processor = ChunkProcessor(chunks=chunks, messages=[])
        result = processor._calculate_usage_per_chunk(chunks=chunks)
        assert result["completion_tokens"] == 5

    def test_no_usage_chunks_leaves_zero(self):
        """Stream with zero usage info → completion_tokens stays 0
        (token_counter fallback will handle it)."""
        chunks = [_make_chunk(content="hi"), _make_chunk(content=" there")]
        processor = ChunkProcessor(chunks=chunks, messages=[])
        result = processor._calculate_usage_per_chunk(chunks=chunks)
        assert result["prompt_tokens"] == 0
        assert result["completion_tokens"] == 0
