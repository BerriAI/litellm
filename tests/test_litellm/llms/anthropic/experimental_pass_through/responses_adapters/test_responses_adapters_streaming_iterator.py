"""
Tests for AnthropicResponsesStreamWrapper
(litellm/llms/anthropic/experimental_pass_through/responses_adapters/streaming_iterator.py)
"""

import asyncio
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../..")))

from litellm.llms.anthropic.experimental_pass_through.responses_adapters.streaming_iterator import (
    AnthropicResponsesStreamWrapper,
)


def _process_all(events: list) -> list:
    wrapper = AnthropicResponsesStreamWrapper(responses_stream=None, model="m")
    for event in events:
        wrapper._process_event(event)
    return list(wrapper._chunk_queue)


def _drain_async(events: list) -> list:
    async def _gen():
        for event in events:
            yield event

    async def _run() -> list:
        wrapper = AnthropicResponsesStreamWrapper(responses_stream=_gen(), model="m")
        return [chunk async for chunk in wrapper]

    return asyncio.run(_run())


class TestMessageStartEmittedExactlyOnce:
    """The ``__anext__`` fallback emits ``message_start`` before consuming the
    stream, so ``_process_event`` must not emit a second one when
    ``response.created`` later arrives. Two ``message_start`` events (byte
    identical, same id) break strict Anthropic SDK clients (e.g. Claude Code)
    with 'Content block is not a thinking block' once thinking blocks follow."""

    def test_response_created_does_not_duplicate_message_start(self):
        chunks = _drain_async(
            [
                {"type": "response.created"},
                {"type": "response.output_text.delta", "item_id": "m1", "delta": "hi"},
            ]
        )
        message_starts = [c for c in chunks if c["type"] == "message_start"]
        assert len(message_starts) == 1

    def test_message_start_is_first_event(self):
        chunks = _drain_async([{"type": "response.created"}])
        assert chunks[0]["type"] == "message_start"


class TestProcessEventResponseCreatedGuard:
    """``_process_event`` must emit ``message_start`` exactly once even if
    ``response.created`` arrives more than once. The guard mirrors the
    ``__anext__`` fallback's ``_sent_message_start`` flag, so a direct caller
    and the async fallback can never double-emit. This also exercises the
    guard's emit-branch, which the async path never reaches because the
    fallback sets the flag before the upstream stream is consumed."""

    def test_first_response_created_emits_message_start(self):
        chunks = _process_all([{"type": "response.created"}])
        assert len(chunks) == 1
        assert chunks[0]["type"] == "message_start"
        assert chunks[0]["message"]["model"] == "m"

    def test_second_response_created_is_skipped(self):
        chunks = _process_all([{"type": "response.created"}, {"type": "response.created"}])
        message_starts = [c for c in chunks if c["type"] == "message_start"]
        assert len(message_starts) == 1


class TestReasoningItemWithoutSummaryText:
    """Regression: a reasoning item whose summary never produces text must not
    surface as a thinking content block.

    OpenAI emits ``response.output_item.added`` with ``type: "reasoning"`` on
    every reasoning turn, but only emits
    ``response.reasoning_summary_text.delta`` when a summary was requested and
    the model actually produced one. Eagerly opening the block on
    ``output_item.added`` left ``{"type": "thinking", "thinking": ""}`` in the
    assistant turn, which clients persist in their session transcript. Replaying
    that transcript against an Anthropic model (what ``claude --resume`` does
    once the resumed session falls back to the default Anthropic model) fails
    with::

        400 invalid_request_error - messages.2.content.0.thinking:
        each thinking block must contain thinking

    So the thinking block is opened on the first non-empty summary delta.
    """

    @staticmethod
    def _gpt_turn(reasoning_summary_deltas: list) -> list:
        return [
            {"type": "response.created"},
            {"type": "response.output_item.added", "item": {"type": "reasoning", "id": "rs_1"}},
            *(
                {"type": "response.reasoning_summary_text.delta", "item_id": "rs_1", "delta": delta}
                for delta in reasoning_summary_deltas
            ),
            {"type": "response.output_item.done", "item": {"type": "reasoning", "id": "rs_1"}},
            {"type": "response.output_item.added", "item": {"type": "message", "id": "msg_1"}},
            {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "Hello"},
            {"type": "response.output_item.done", "item": {"type": "message", "id": "msg_1"}},
        ]

    def test_reasoning_without_summary_emits_no_thinking_block(self):
        chunks = _drain_async(self._gpt_turn(reasoning_summary_deltas=[]))

        assert not [
            c for c in chunks if c["type"] == "content_block_start" and c["content_block"]["type"] == "thinking"
        ]
        assert [(c["type"], c.get("index")) for c in chunks[1:]] == [
            ("content_block_start", 0),
            ("content_block_delta", 0),
            ("content_block_stop", 0),
        ]
        assert chunks[1]["content_block"] == {"type": "text", "text": ""}

    def test_reasoning_with_only_empty_summary_deltas_emits_no_thinking_block(self):
        chunks = _drain_async(self._gpt_turn(reasoning_summary_deltas=["", ""]))

        assert not [c for c in chunks if c["type"] == "content_block_delta" and c["delta"]["type"] == "thinking_delta"]
        assert not [
            c for c in chunks if c["type"] == "content_block_start" and c["content_block"]["type"] == "thinking"
        ]

    def test_reasoning_with_summary_text_still_emits_a_thinking_block(self):
        chunks = _drain_async(self._gpt_turn(reasoning_summary_deltas=["Weigh", "ing options"]))

        assert [(c["type"], c.get("index")) for c in chunks[1:]] == [
            ("content_block_start", 0),
            ("content_block_delta", 0),
            ("content_block_delta", 0),
            ("content_block_stop", 0),
            ("content_block_start", 1),
            ("content_block_delta", 1),
            ("content_block_stop", 1),
        ]
        assert chunks[1]["content_block"] == {"type": "thinking", "thinking": "", "signature": ""}
        assert "".join(c["delta"]["thinking"] for c in chunks[2:4]) == "Weighing options"

    def test_the_reasoning_item_id_is_never_streamed_as_a_signature(self):
        """A stand-in signature would be replayed as a real one, so none is ever sent."""
        chunks = _drain_async(self._gpt_turn(reasoning_summary_deltas=["Weighing options"]))

        assert not [c for c in chunks if c.get("delta", {}).get("type") == "signature_delta"]


class TestToolUseBlockClosedExactlyOnce:
    """Regression for https://github.com/BerriAI/litellm/issues/37273.

    With ``custom_llm_provider: openai`` + ``use_chat_completions_api: true``,
    ``/v1/messages`` streams through ``LiteLLMCompletionStreamingIterator``,
    which ends a tool-call turn with two ``response.output_item.done`` events:
    one for the function_call item (id = call_id) and one for a synthetic
    message item whose id is the upstream chatcmpl id and was never opened as a
    content block. Resolving that unknown item id to ``_current_block_index``
    closed the tool_use block a second time::

        content_block_start[0](tool_use) -> content_block_stop[0]
        -> content_block_stop[0] -> message_delta(stop_reason=tool_use)

    Anthropic SDK clients (e.g. Claude Code) materialize one tool_use block per
    ``content_block_stop``, so the tool executed twice. An ``output_item.done``
    for an item that never opened a block must emit nothing.
    """

    @staticmethod
    def _chat_completions_bridge_tool_turn() -> list[dict[str, object]]:
        return [
            {"type": "response.created"},
            {
                "type": "response.output_item.added",
                "item": {"type": "function_call", "id": "call_1", "call_id": "call_1", "name": "get_weather"},
            },
            {"type": "response.function_call_arguments.delta", "item_id": "call_1", "delta": '{"city": "'},
            {"type": "response.function_call_arguments.delta", "item_id": "call_1", "delta": 'Tokyo"}'},
            {
                "type": "response.function_call_arguments.done",
                "item_id": "call_1",
                "arguments": '{"city": "Tokyo"}',
            },
            {
                "type": "response.output_item.done",
                "item": {"type": "function_call", "id": "call_1", "call_id": "call_1", "status": "completed"},
            },
            {
                "type": "response.output_item.done",
                "item": {"type": "message", "id": "chatcmpl-123", "status": "completed"},
            },
        ]

    def test_one_content_block_stop_per_content_block_start(self):
        chunks = _drain_async(self._chat_completions_bridge_tool_turn())

        starts = [c["index"] for c in chunks if c["type"] == "content_block_start"]
        stops = [c["index"] for c in chunks if c["type"] == "content_block_stop"]
        assert starts == [0]
        assert stops == [0]

    def test_tool_turn_event_order(self):
        chunks = _drain_async(self._chat_completions_bridge_tool_turn())

        assert [(c["type"], c.get("index")) for c in chunks] == [
            ("message_start", None),
            ("content_block_start", 0),
            ("content_block_delta", 0),
            ("content_block_delta", 0),
            ("content_block_stop", 0),
        ]
        assert chunks[1]["content_block"] == {
            "type": "tool_use",
            "id": "call_1",
            "name": "get_weather",
            "input": {},
        }


class TestProcessEventTextDeltaWithoutOutputItemAdded:
    """Streams that skip response.output_item.added (e.g. LMStudio) must still
    open a text block before any delta and never emit index -1."""

    def test_process_event_synthesizes_content_block_start_before_delta(self):
        chunks = _process_all(
            [
                {"type": "response.output_text.delta", "item_id": "i1", "delta": "Hel"},
                {"type": "response.output_text.delta", "item_id": "i1", "delta": "lo"},
            ]
        )
        assert [c["type"] for c in chunks] == [
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
        ]
        assert chunks[0]["content_block"] == {"type": "text", "text": ""}
        assert [c["index"] for c in chunks] == [0, 0, 0]
        assert chunks[1]["delta"] == {"type": "text_delta", "text": "Hel"}

    def test_process_event_delta_without_item_id_never_yields_negative_index(self):
        chunks = _process_all([{"type": "response.output_text.delta", "delta": "Hi"}])
        assert [(c["type"], c["index"]) for c in chunks] == [
            ("content_block_start", 0),
            ("content_block_delta", 0),
        ]

    def test_process_event_unregistered_item_id_opens_new_text_block(self):
        chunks = _process_all(
            [
                {
                    "type": "response.output_item.added",
                    "item": {"type": "reasoning", "id": "rs_1"},
                },
                {"type": "response.reasoning_summary_text.delta", "item_id": "rs_1", "delta": "hm"},
                {"type": "response.output_text.delta", "item_id": "m1", "delta": "Hi"},
            ]
        )
        assert chunks[2]["type"] == "content_block_start"
        assert chunks[2]["content_block"] == {"type": "text", "text": ""}
        assert [c["index"] for c in chunks[2:]] == [1, 1]

    def test_process_event_registered_item_id_does_not_synthesize_start(self):
        chunks = _process_all(
            [
                {
                    "type": "response.output_item.added",
                    "item": {"type": "message", "id": "m1"},
                },
                {"type": "response.output_text.delta", "item_id": "m1", "delta": "Hi"},
            ]
        )
        assert [(c["type"], c["index"]) for c in chunks] == [
            ("content_block_start", 0),
            ("content_block_delta", 0),
        ]


class TestResponseCompletedUsage:
    """The Anthropic ``message_delta`` usage must report cache reads/writes and
    exclude them from ``input_tokens``, so spend is not billed at the uncached
    input rate."""

    def test_response_completed_usage_carries_cache_tokens(self):
        from litellm.types.llms.openai import ResponseAPIUsage

        response = SimpleNamespace(
            status="completed",
            output=[],
            usage=ResponseAPIUsage(
                input_tokens=4017,
                input_tokens_details={"cached_tokens": 4004, "cache_write_tokens": 10},
                output_tokens=5,
                total_tokens=4022,
            ),
        )
        chunks = _process_all([{"type": "response.completed", "response": response}])
        message_delta = next(c for c in chunks if c["type"] == "message_delta")
        assert message_delta["usage"] == {
            "input_tokens": 3,
            "output_tokens": 5,
            "cache_creation_input_tokens": 10,
            "cache_read_input_tokens": 4004,
        }
