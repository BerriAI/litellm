"""
Unit tests for ``_transform_responses_api_input_item_to_chat_completion_message``
handling of ``ResponseReasoningItemParam`` (``type: "reasoning"``).

Covers the Responses-API path of BerriAI/litellm#26395 — without the fix,
prior-turn reasoning items pollute the prompt as visible content and leave
the adjacent assistant message without ``reasoning_content``, which DeepSeek
V4 rejects with HTTP 400 ``reasoning_content must be passed back``.
"""

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)


def _transform_item(item):
    return LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
        input_item=item
    )


class TestReasoningInputItemHandler:
    """Reasoning items are dropped from the Chat-Completion message stream."""

    def test_reasoning_item_with_output_text_dropped(self):
        """Standard Responses-API reasoning item shape → []."""
        item = {
            "type": "reasoning",
            "id": "rs_abc",
            "summary": [],
            "content": [{"type": "output_text", "text": "step 1: think about X"}],
        }
        assert _transform_item(item) == []

    def test_reasoning_item_with_string_content_dropped(self):
        """Variant: reasoning content as a plain string → []."""
        item = {"type": "reasoning", "id": "rs_1", "content": "step 1: ..."}
        assert _transform_item(item) == []

    def test_reasoning_item_with_summary_only_dropped(self):
        """SDK 0.17 form: reasoning carried in summary list, no content → []."""
        item = {
            "type": "reasoning",
            "id": "rs_2",
            "summary": [{"type": "summary_text", "text": "..."}],
        }
        assert _transform_item(item) == []

    def test_reasoning_item_empty_dropped(self):
        """Reasoning item with neither content nor summary still drops cleanly."""
        item = {"type": "reasoning", "id": "rs_3"}
        assert _transform_item(item) == []


class TestNonReasoningInputItemUnchanged:
    """Non-reasoning items still flow through the existing branches."""

    def test_user_message_unchanged(self):
        item = {"role": "user", "content": "hello"}
        out = _transform_item(item)
        assert len(out) == 1
        assert out[0].get("role") == "user"

    def test_assistant_message_unchanged(self):
        item = {"role": "assistant", "content": "hi"}
        out = _transform_item(item)
        assert len(out) == 1
        assert out[0].get("role") == "assistant"

    def test_none_content_still_returns_empty(self):
        """Pre-existing behavior: None content → [] (unchanged by this fix)."""
        item = {"role": "user", "content": None}
        assert _transform_item(item) == []
