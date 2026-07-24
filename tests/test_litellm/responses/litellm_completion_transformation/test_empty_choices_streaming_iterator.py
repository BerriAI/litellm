"""
Regression tests for LIT-4767.

When an upstream OpenAI-compatible provider emits a chunk with ``choices: []``
(the trailing usage-only chunk every provider sends when ``include_usage`` is
set, or Azure's leading ``prompt_filter_results`` chunk), the Responses bridge
iterator used to index ``choices[0]`` unguarded and die with
``IndexError: list index out of range``, killing the whole stream.

The empty-choices chunk must be tolerated without crashing, and the usage it
carries must still reach ``response.completed``.
"""

from unittest.mock import AsyncMock

from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.types.llms.openai import ResponsesAPIStreamEvents
from litellm.types.utils import (
    Delta,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)


def _iterator() -> LiteLLMCompletionStreamingIterator:
    return LiteLLMCompletionStreamingIterator(
        model="gpt-4o",
        litellm_custom_stream_wrapper=AsyncMock(),
        request_input="hi",
        responses_api_request={},
        custom_llm_provider="openai",
    )


def _empty_choices_usage_chunk() -> ModelResponseStream:
    chunk = ModelResponseStream(id="chunk-usage", model="gpt-4o", choices=[])
    chunk.usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    return chunk


def test_ensure_output_item_for_empty_choices_chunk_does_not_crash():
    """First chunk with no choices must not raise (traceback frame in the ticket)."""
    iterator = _iterator()
    # Would raise IndexError before the fix.
    assert iterator._ensure_output_item_for_chunk(_empty_choices_usage_chunk()) is None
    assert iterator.sent_output_item_added_event is False


def test_transform_empty_choices_chunk_returns_no_delta():
    """The mid/trailing usage chunk flows through transform without crashing."""
    iterator = _iterator()
    # Would raise IndexError in _get_delta_string_from_streaming_choices before the fix.
    assert iterator._transform_chat_completion_chunk_to_response_api_chunk(_empty_choices_usage_chunk()) is None


def test_is_reasoning_end_false_for_empty_choices_chunk():
    iterator = _iterator()
    assert iterator._is_reasoning_end(_empty_choices_usage_chunk()) is False


def test_empty_choices_usage_chunk_still_reaches_response_completed():
    """End-to-end: a text chunk followed by a choices=[] usage chunk must emit
    response.completed carrying the usage rather than dying mid-stream."""

    class _SyncWrapper:
        def __init__(self, chunks):
            self._it = iter(chunks)
            self.logging_obj = None
            self.stream_options = {"include_usage": True}

        def __next__(self):
            return next(self._it)

    text_chunk = ModelResponseStream(
        id="chunk-1",
        model="gpt-4o",
        choices=[StreamingChoices(index=0, delta=Delta(role="assistant", content="Hi"), finish_reason=None)],
    )
    iterator = _iterator()
    iterator.litellm_logging_obj = None
    iterator.litellm_custom_stream_wrapper = _SyncWrapper([text_chunk, _empty_choices_usage_chunk()])

    events = list(iterator)

    completed = [e for e in events if getattr(e, "type", None) == ResponsesAPIStreamEvents.RESPONSE_COMPLETED]
    assert len(completed) == 1
    assert completed[0].response.usage is not None
    assert completed[0].response.usage.total_tokens == 15
