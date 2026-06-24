"""
Regression tests for https://github.com/BerriAI/litellm/issues/26153

``stream_chunk_builder`` used to leave ``usage.server_tool_use`` as a plain
``dict`` when reconstructing a streaming response. Downstream cost-calculation
code (``StandardBuiltInToolCostTracking.response_object_includes_web_search_call``
and ``get_cost_for_anthropic_web_search``) accesses
``usage.server_tool_use.web_search_requests`` as an attribute, which raised
``AttributeError: 'dict' object has no attribute 'web_search_requests'``.

These tests reconstruct streaming chunks for an Anthropic-style web_search
response and assert:

1. ``stream_chunk_builder`` returns ``ServerToolUse`` (not ``dict``) for
   ``usage.server_tool_use``.
2. ``completion_cost`` runs end-to-end on the rebuilt response without
   raising ``AttributeError``.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm import completion_cost, stream_chunk_builder
from litellm.types.utils import (
    Delta,
    ModelResponseStream,
    ServerToolUse,
    StreamingChoices,
    Usage,
)


def _make_text_chunk(text: str) -> ModelResponseStream:
    return ModelResponseStream(
        id="chatcmpl-test-26153",
        created=1700000000,
        model="claude-3-haiku-20240307",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(role="assistant", content=text),
            )
        ],
    )


def _make_finish_chunk_with_usage_dict_server_tool_use() -> ModelResponseStream:
    """Final chunk where server_tool_use is a *dict* — reproduces the bug shape."""
    return ModelResponseStream(
        id="chatcmpl-test-26153",
        created=1700000000,
        model="claude-3-haiku-20240307",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(),
            )
        ],
        usage=Usage(
            prompt_tokens=42,
            completion_tokens=11,
            total_tokens=53,
            # NOTE: passed as a dict on purpose — this is the shape that
            # historically slipped through stream_chunk_builder unchanged.
            server_tool_use={"web_search_requests": 3},
        ),
    )


def test_stream_chunk_builder_coerces_server_tool_use_to_pydantic():
    """
    Regression: stream_chunk_builder must produce ServerToolUse, not dict.
    """
    chunks = [
        _make_text_chunk("Otters "),
        _make_text_chunk("are great."),
        _make_finish_chunk_with_usage_dict_server_tool_use(),
    ]

    rebuilt = stream_chunk_builder(chunks)

    assert rebuilt is not None
    assert rebuilt.usage is not None  # type: ignore[attr-defined]
    server_tool_use = rebuilt.usage.server_tool_use  # type: ignore[attr-defined]

    assert (
        server_tool_use is not None
    ), "server_tool_use should be carried through from the final chunk"
    assert isinstance(server_tool_use, ServerToolUse), (
        f"expected ServerToolUse, got {type(server_tool_use).__name__}: "
        f"{server_tool_use!r}"
    )
    # Attribute access must not raise (this is exactly what was broken).
    assert server_tool_use.web_search_requests == 3


def test_completion_cost_does_not_raise_on_streaming_web_search_response():
    """
    Regression: completion_cost(...) must not raise AttributeError when the
    response was reconstructed by stream_chunk_builder from a streaming
    Anthropic web_search call.
    """
    chunks = [
        _make_text_chunk("hello"),
        _make_finish_chunk_with_usage_dict_server_tool_use(),
    ]

    rebuilt = stream_chunk_builder(chunks)
    assert rebuilt is not None

    # The exact dollar amount depends on the model-pricing table; what matters
    # for this regression is that it does NOT raise AttributeError on
    # `dict has no attribute 'web_search_requests'`.
    try:
        cost = completion_cost(completion_response=rebuilt)
    except AttributeError as e:  # pragma: no cover - regression guard
        pytest.fail(
            "completion_cost raised AttributeError after stream_chunk_builder "
            f"(issue #26153 regression): {e}"
        )

    assert isinstance(cost, (int, float))
