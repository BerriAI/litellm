"""Regression test for #30761 — the Anthropic streaming bridge crashed with
IndexError when an upstream OpenAI/Azure-compatible chunk had choices=[]
(usage-only / keepalive chunk)."""

import os
import sys
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.types.utils import Delta, StreamingChoices, Usage


def _content_chunk(text: str, finish_reason: Optional[str] = None) -> MagicMock:
    chunk = MagicMock()
    chunk.choices = [
        StreamingChoices(
            finish_reason=finish_reason,
            index=0,
            delta=Delta(content=text),
            logprobs=None,
        )
    ]
    chunk.usage = None
    chunk._hidden_params = {}
    return chunk


def _usage_only_chunk() -> MagicMock:
    # OpenAI include_usage trailing chunk: empty choices, usage set
    chunk = MagicMock()
    chunk.choices = []
    chunk.usage = Usage(prompt_tokens=5, completion_tokens=3, total_tokens=8)
    chunk._hidden_params = {}
    return chunk


def test_sync_stream_survives_empty_choices_usage_chunk():
    chunks = [
        _content_chunk("hi"),
        _content_chunk("", finish_reason="stop"),
        _usage_only_chunk(),
    ]
    wrapper = AnthropicStreamWrapper(completion_stream=iter(chunks), model="claude-x")
    events = list(wrapper)  # used to raise IndexError mid-stream
    assert events


@pytest.mark.asyncio
async def test_async_stream_survives_empty_choices_usage_chunk():
    class _AsyncStream:
        def __init__(self, items):
            self._it = iter(items)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._it)
            except StopIteration:
                raise StopAsyncIteration

    chunks = [
        _content_chunk("hi"),
        _content_chunk("", finish_reason="stop"),
        _usage_only_chunk(),
    ]
    wrapper = AnthropicStreamWrapper(
        completion_stream=_AsyncStream(chunks), model="claude-x"
    )
    events = [e async for e in wrapper]
    assert events
