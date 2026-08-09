"""
Unit tests for preserving prior-turn ``reasoning`` input items when the
Responses API is bridged to chat completions.

Without this handling, a ``ResponseReasoningItemParam`` falls through to the
generic message branch, polluting the prompt as visible assistant ``content``
or being silently dropped. Chat-completions providers such as DeepSeek V4 and
Kimi K2.6 require the chain-of-thought to be replayed as ``reasoning_content``
on an assistant message.
"""

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)


def _transform_item(item):
    return LiteLLMCompletionResponsesConfig._transform_responses_api_input_item_to_chat_completion_message(
        input_item=item
    )


def _transform_input(input_items):
    return LiteLLMCompletionResponsesConfig._transform_response_input_param_to_chat_completion_message(
        input=input_items
    )


class TestReasoningInputItemHandler:
    """Reasoning input items map to assistant ``reasoning_content``."""

    def test_reasoning_item_with_output_text_content(self):
        """Standard Responses-API reasoning item with output_text blocks."""
        item = {
            "type": "reasoning",
            "id": "rs_abc",
            "summary": [],
            "content": [{"type": "output_text", "text": "step 1: think about X"}],
        }
        messages = _transform_item(item)
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] is None
        assert messages[0]["reasoning_content"] == "step 1: think about X"

    def test_reasoning_item_with_string_content(self):
        """Variant: reasoning content as a plain string."""
        item = {"type": "reasoning", "id": "rs_1", "content": "step 1: ..."}
        messages = _transform_item(item)
        assert messages[0]["reasoning_content"] == "step 1: ..."

    def test_reasoning_item_with_summary_only(self):
        """SDK form: reasoning carried in summary list, no content."""
        item = {
            "type": "reasoning",
            "id": "rs_2",
            "summary": [{"type": "summary_text", "text": "..."}],
        }
        messages = _transform_item(item)
        assert messages[0]["reasoning_content"] == "..."

    def test_reasoning_item_with_encrypted_content_only_dropped(self):
        """Opaque encrypted reasoning cannot be forwarded to chat completions."""
        item = {"type": "reasoning", "id": "rs_3", "encrypted_content": "opaque-blob"}
        assert _transform_item(item) == []

    def test_reasoning_item_empty_dropped(self):
        """Reasoning item with neither content nor summary drops cleanly."""
        assert _transform_item({"type": "reasoning", "id": "rs_4"}) == []


class TestReasoningInputItemMerging:
    """Standalone reasoning messages merge into the following assistant turn."""

    def test_reasoning_merged_into_following_assistant_message(self):
        """Reasoning + assistant answer become one assistant message."""
        messages = _transform_input(
            [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "content": [{"type": "output_text", "text": "secret reasoning"}],
                },
                {"type": "message", "role": "assistant", "content": "The answer."},
            ]
        )
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] == "The answer."
        assert messages[0]["reasoning_content"] == "secret reasoning"

    def test_reasoning_preserved_when_followed_by_user_message(self):
        """Stateless chain: reasoning + user prompt keeps the reasoning turn."""
        messages = _transform_input(
            [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "content": [{"type": "output_text", "text": "secret BLUEBERRY"}],
                },
                {"role": "user", "content": "What is the secret word?"},
            ]
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"] is None
        assert messages[0]["reasoning_content"] == "secret BLUEBERRY"
        assert messages[1]["role"] == "user"

    def test_reasoning_merged_into_function_call_assistant(self):
        """Reasoning + function_call becomes one assistant tool-call message."""
        messages = _transform_input(
            [
                {
                    "type": "reasoning",
                    "id": "rs_1",
                    "content": [{"type": "output_text", "text": "I should look this up"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": '{"cwe": "79"}',
                },
            ]
        )
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["reasoning_content"] == "I should look this up"
        assert len(messages[0]["tool_calls"]) == 1

    def test_reasoning_merged_into_assistant_with_existing_reasoning_content(self):
        """Old reasoning precedes existing reasoning on the target assistant turn."""
        messages = LiteLLMCompletionResponsesConfig._merge_reasoning_only_assistant_messages(
            [
                {"role": "assistant", "content": None, "reasoning_content": "old reasoning"},
                {"role": "assistant", "content": "The answer.", "reasoning_content": "new reasoning"},
            ]
        )
        assert len(messages) == 1
        assert messages[0]["content"] == "The answer."
        assert messages[0]["reasoning_content"] == "old reasoning\nnew reasoning"


class TestNonReasoningInputItemUnchanged:
    """Non-reasoning items still flow through the existing branches."""

    def test_user_message_unchanged(self):
        item = {"role": "user", "content": "hello"}
        out = _transform_item(item)
        assert len(out) == 1
        assert out[0]["role"] == "user"

    def test_assistant_message_unchanged(self):
        item = {"role": "assistant", "content": "hi"}
        out = _transform_item(item)
        assert len(out) == 1
        assert out[0]["role"] == "assistant"
        assert out[0]["content"] == "hi"
