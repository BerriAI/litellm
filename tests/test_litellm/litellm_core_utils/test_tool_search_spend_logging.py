"""
Integration / regression tests for Anthropic tool-search (`tool_reference`)
content blocks on the cost-calculation and streaming-assembly paths used by
Claude Code.

Claude Code's tool-search feature emits assistant content blocks of the form
``{"type": "tool_reference", "tool_name": ...}`` -- a lightweight pointer to a
deferred tool. Before the fix, `token_counter` did not recognise this block
type and raised ``Invalid content item type: tool_reference``.

Why this matters (the bug these tests guard against):

  * On the cost path, that exception propagates out of ``completion_cost`` ->
    ``response_cost_calculator``. The proxy logging layer catches it and nulls
    ``response_cost``; the spend-tracking callback then skips the request, so
    the entire SpendLogs row is dropped. The request succeeds for the caller
    but the spend is silently never recorded -- a cost undercount on ALL
    tool-search traffic.

  * On the streaming-assembly path, ``stream_chunk_builder`` recomputes the
    prompt tokens from the request messages when the provider stream does not
    carry usage. The same exception there was swallowed and prompt tokens
    silently collapsed to 0 -- a quieter undercount of the same traffic.

These tests exercise the real public entry points (not the private
``_count_content_list`` helper) so the whole chain is covered end to end.
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm import stream_chunk_builder
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

ANTHROPIC_MODEL = "anthropic/claude-sonnet-4-5-20250929"

# Mirrors a Claude Code tool-search turn: a normal text block followed by a
# `tool_reference` pointer to a deferred tool.
TOOL_SEARCH_MESSAGES = [
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Let me look up the right tool."},
            {"type": "tool_reference", "tool_name": "search_knowledge_base"},
        ],
    }
]


def test_completion_cost_with_tool_reference_records_spend():
    """
    ``completion_cost`` must return a real, positive cost for messages that
    contain a tool-search ``tool_reference`` block.

    This is the exact chain that fails on the streaming anthropic_messages
    proxy path: before the fix ``completion_cost`` raised, the logging layer
    caught the exception and set ``response_cost = None``, and the spend
    callback then dropped the SpendLogs row. A positive cost here means the
    row is recorded instead of silently dropped.
    """
    cost = litellm.completion_cost(model=ANTHROPIC_MODEL, messages=TOOL_SEARCH_MESSAGES)

    assert cost is not None, "response_cost is None -> SpendLogs row would be dropped"
    assert cost > 0, f"Expected a positive cost for tool-search traffic, got {cost}"


def test_completion_cost_with_empty_tool_name_records_spend():
    """A ``tool_reference`` with an empty/missing ``tool_name`` must also cost
    out cleanly rather than raising and nulling the spend."""
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_reference", "tool_name": ""}],
        }
    ]

    cost = litellm.completion_cost(model=ANTHROPIC_MODEL, messages=messages)

    assert cost is not None
    assert cost >= 0


def test_stream_chunk_builder_counts_prompt_tokens_for_tool_reference():
    """
    On the streaming-assembly path used by Claude Code, when the provider
    stream carries no prompt-token usage, ``stream_chunk_builder`` recomputes
    prompt tokens from the request messages via ``token_counter``.

    With a ``tool_reference`` block in those messages the count must be
    positive. Before the fix the underlying ``token_counter`` call raised and
    the assembler swallowed it, collapsing ``prompt_tokens`` to 0 -- a silent
    undercount of every tool-search request.
    """
    model = "claude-sonnet-4-5-20250929"
    # Chunks deliberately carry no usage, forcing the prompt-token fallback.
    chunks = [
        ModelResponseStream(
            id="chatcmpl-tool-search",
            created=1700000000,
            model=model,
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(content="Searching...", role="assistant"),
                )
            ],
        ),
        ModelResponseStream(
            id="chatcmpl-tool-search",
            created=1700000000,
            model=model,
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    finish_reason="stop", index=0, delta=Delta(content="")
                ),
            ],
        ),
    ]

    response = stream_chunk_builder(chunks, messages=TOOL_SEARCH_MESSAGES)

    assert response is not None
    assert (
        response.usage.prompt_tokens > 0
    ), "prompt_tokens collapsed to 0 -> tool-search traffic silently undercounted"
