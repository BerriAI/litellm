import os
import sys
from typing import List, Optional
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.types.utils import Delta, StreamingChoices


def _make_chunk(
    *,
    content: Optional[str] = None,
    thinking: Optional[str] = None,
    signature: Optional[str] = None,
    finish_reason: Optional[str] = None,
) -> MagicMock:
    """Create a minimal streaming chunk matching Databricks->LiteLLM shape."""
    thinking_blocks = None
    if thinking is not None or signature is not None:
        thinking_blocks = [
            {
                "type": "thinking",
                "thinking": thinking,
                "signature": signature,
            }
        ]

    chunk = MagicMock()
    chunk.choices = [
        StreamingChoices(
            finish_reason=finish_reason,
            index=0,
            delta=Delta(
                content=content,
                role="assistant",
                tool_calls=None,
                thinking_blocks=thinking_blocks,
                reasoning_content=thinking,
            ),
            logprobs=None,
        )
    ]
    chunk.usage = None
    chunk._hidden_params = {}
    return chunk


def _collect_events_sync(wrapper: AnthropicStreamWrapper) -> List[dict]:
    return [event for event in wrapper]


def _build_databricks_like_stream() -> List[MagicMock]:
    """
    Mimic the real failing sequence:
    thinking chunks -> empty thinking chunk -> signature chunk -> text chunks -> stop.
    """
    return [
        _make_chunk(content="", thinking="The user is asking", signature=""),
        _make_chunk(content="", thinking=' "', signature=""),
        _make_chunk(content="", thinking="Hello", signature=""),
        _make_chunk(content="", thinking=', who are you?" in Chinese.', signature=""),
        _make_chunk(content="", thinking="", signature=""),
        _make_chunk(content="", thinking="", signature="sig_payload"),
        _make_chunk(content="你"),
        _make_chunk(content="好！我是 "),
        _make_chunk(content="", finish_reason="stop"),
    ]


def test_thinking_first_chunk_should_not_create_empty_text_block():
    wrapper = AnthropicStreamWrapper(
        completion_stream=iter(_build_databricks_like_stream()),
        model="claude-4-6-opus",
    )
    events = _collect_events_sync(wrapper)

    content_block_starts = [
        e
        for e in events
        if isinstance(e, dict) and e.get("type") == "content_block_start"
    ]
    assert content_block_starts, "Expected content_block_start events"

    first_start = content_block_starts[0]
    assert first_start["index"] == 0
    assert first_start["content_block"]["type"] == "thinking"


def test_thinking_content_block_start_payload_should_be_empty():
    wrapper = AnthropicStreamWrapper(
        completion_stream=iter(_build_databricks_like_stream()),
        model="claude-4-6-opus",
    )
    events = _collect_events_sync(wrapper)

    thinking_start = next(
        e
        for e in events
        if isinstance(e, dict)
        and e.get("type") == "content_block_start"
        and isinstance(e.get("content_block"), dict)
        and e["content_block"].get("type") == "thinking"
    )

    assert thinking_start["content_block"]["thinking"] == ""
    assert thinking_start["content_block"]["signature"] == ""


def test_thinking_block_should_not_emit_text_delta():
    wrapper = AnthropicStreamWrapper(
        completion_stream=iter(_build_databricks_like_stream()),
        model="claude-4-6-opus",
    )
    events = _collect_events_sync(wrapper)

    thinking_start = next(
        e
        for e in events
        if isinstance(e, dict)
        and e.get("type") == "content_block_start"
        and isinstance(e.get("content_block"), dict)
        and e["content_block"].get("type") == "thinking"
    )
    thinking_index = thinking_start["index"]

    thinking_block_deltas = [
        e["delta"]
        for e in events
        if isinstance(e, dict)
        and e.get("type") == "content_block_delta"
        and e.get("index") == thinking_index
    ]

    delta_types = [delta.get("type") for delta in thinking_block_deltas]
    assert "text_delta" not in delta_types
