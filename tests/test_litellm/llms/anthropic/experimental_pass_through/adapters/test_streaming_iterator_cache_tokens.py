"""
Regression tests for prompt-cache token surfacing in AnthropicStreamWrapper.

When a Claude model is routed through an OpenAI-format upstream (e.g. OpenRouter)
and called via the Anthropic-native ``/v1/messages`` streaming endpoint, the
provider sends ``finish_reason`` and the ``include_usage`` usage in two separate
chunks. The wrapper holds the stop_reason chunk and merges the trailing
usage-only chunk via ``_merge_usage_into_held_stop_reason_chunk``.

For OpenAI-format upstreams the cache-read count lives in
``usage.prompt_tokens_details.cached_tokens`` (the private
``_cache_read_input_tokens`` attribute, set only on the Anthropic-native path,
stays 0). The merge path used to emit ``cache_read_input_tokens`` solely from
that private attribute, so the final ``message_delta`` dropped the cache-hit
count even though the uncached ``input_tokens`` had already been reduced by it.
These tests pin the merged ``message_delta`` to carry ``cache_read_input_tokens``
from ``cached_tokens``, matching the non-streaming path.
"""

import os
import sys
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.types.utils import (
    Delta,
    PromptTokensDetailsWrapper,
    StreamingChoices,
    Usage,
)


def _make_chunk(
    delta: Delta,
    finish_reason: Optional[str] = None,
    usage: Optional[Usage] = None,
) -> MagicMock:
    chunk = MagicMock()
    chunk.choices = [
        StreamingChoices(
            finish_reason=finish_reason,
            index=0,
            delta=delta,
            logprobs=None,
        )
    ]
    chunk.usage = usage
    chunk._hidden_params = {}
    return chunk


def _final_message_delta(events: List) -> dict:
    deltas = [
        e for e in events if isinstance(e, dict) and e.get("type") == "message_delta"
    ]
    assert deltas, f"no message_delta emitted; events: {events}"
    return deltas[-1]


def _openai_format_cache_hit_chunks() -> list:
    """text -> finish_reason (no usage) -> usage-only chunk (OpenAI-format).

    Mirrors OpenRouter streaming with stream_options.include_usage: the cache
    read count is in prompt_tokens_details.cached_tokens and the private
    _cache_read_input_tokens attribute is left at its default 0.
    """
    text_chunk = _make_chunk(Delta(content="OK", role="assistant", tool_calls=None))
    finish_chunk = _make_chunk(
        Delta(content=None, role="assistant", tool_calls=None),
        finish_reason="stop",
    )
    usage = Usage(
        prompt_tokens=7009,
        completion_tokens=4,
        total_tokens=7013,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=7001),
    )
    assert usage._cache_read_input_tokens == 0  # the OpenAI-format precondition
    usage_chunk = _make_chunk(
        Delta(content=None, role="assistant", tool_calls=None),
        finish_reason=None,
        usage=usage,
    )
    return [text_chunk, finish_chunk, usage_chunk]


def test_sync_merge_surfaces_cache_read_from_cached_tokens():
    wrapper = AnthropicStreamWrapper(
        completion_stream=iter(_openai_format_cache_hit_chunks()),
        model="anthropic/claude-haiku-4.5",
    )

    delta = _final_message_delta(list(wrapper))
    usage = delta["usage"]
    assert usage["cache_read_input_tokens"] == 7001
    assert usage["input_tokens"] == 8  # 7009 prompt - 7001 cached
    assert usage["output_tokens"] == 4


@pytest.mark.asyncio
async def test_async_merge_surfaces_cache_read_from_cached_tokens():
    chunks = _openai_format_cache_hit_chunks()

    async def mock_stream():
        for c in chunks:
            yield c

    wrapper = AnthropicStreamWrapper(
        completion_stream=mock_stream(),
        model="anthropic/claude-haiku-4.5",
    )

    events = []
    async for event in wrapper:
        events.append(event)

    usage = _final_message_delta(events)["usage"]
    assert usage["cache_read_input_tokens"] == 7001
    assert usage["input_tokens"] == 8
    assert usage["output_tokens"] == 4


def test_sync_merge_prefers_explicit_cache_read_attr():
    """Anthropic-native path sets the private _cache_read_input_tokens; it must
    still win so this fix does not regress native passthrough."""
    text_chunk = _make_chunk(Delta(content="OK", role="assistant", tool_calls=None))
    finish_chunk = _make_chunk(
        Delta(content=None, role="assistant", tool_calls=None),
        finish_reason="stop",
    )
    usage = Usage(
        prompt_tokens=5000,
        completion_tokens=4,
        total_tokens=5004,
        cache_read_input_tokens=4096,
    )
    assert usage._cache_read_input_tokens == 4096
    usage_chunk = _make_chunk(
        Delta(content=None, role="assistant", tool_calls=None),
        usage=usage,
    )

    wrapper = AnthropicStreamWrapper(
        completion_stream=iter([text_chunk, finish_chunk, usage_chunk]),
        model="anthropic/claude-haiku-4.5",
    )

    usage_out = _final_message_delta(list(wrapper))["usage"]
    assert usage_out["cache_read_input_tokens"] == 4096


def test_sync_merge_omits_cache_read_when_no_cache_hit():
    """No cache hit (cached_tokens == 0) must not emit cache_read_input_tokens."""
    text_chunk = _make_chunk(Delta(content="OK", role="assistant", tool_calls=None))
    finish_chunk = _make_chunk(
        Delta(content=None, role="assistant", tool_calls=None),
        finish_reason="stop",
    )
    usage = Usage(prompt_tokens=50, completion_tokens=4, total_tokens=54)
    usage_chunk = _make_chunk(
        Delta(content=None, role="assistant", tool_calls=None),
        usage=usage,
    )

    wrapper = AnthropicStreamWrapper(
        completion_stream=iter([text_chunk, finish_chunk, usage_chunk]),
        model="anthropic/claude-haiku-4.5",
    )

    usage_out = _final_message_delta(list(wrapper))["usage"]
    assert "cache_read_input_tokens" not in usage_out
    assert usage_out["input_tokens"] == 50
