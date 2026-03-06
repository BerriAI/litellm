"""
Tests for thinking_blocks ↔ reasoning items passthrough in the
Chat Completions → Responses API bridge.

Covers:
- _extract_reasoning_input_items_from_thinking_blocks()
- convert_chat_completion_messages_to_responses_api() with thinking_blocks
- _convert_response_output_to_choices() with encrypted_content
- transform_request() auto-injection of include: ["reasoning.encrypted_content"]
"""

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from litellm.completion_extras.litellm_responses_transformation.transformation import (
    LiteLLMResponsesTransformationHandler,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def handler() -> LiteLLMResponsesTransformationHandler:
    return LiteLLMResponsesTransformationHandler()


# ---------------------------------------------------------------------------
# _extract_reasoning_input_items_from_thinking_blocks
# ---------------------------------------------------------------------------

class TestExtractReasoningInputItems:
    """Unit tests for the helper that converts thinking_blocks → reasoning items."""

    def test_single_thinking_block_with_encrypted_content(self) -> None:
        msg: Dict[str, Any] = {
            "role": "assistant",
            "content": "Hello",
            "thinking_blocks": [
                {
                    "type": "thinking",
                    "thinking": "I should greet the user.",
                    "encrypted_content": "opaque_blob_123",
                    "id": "rs_abc",
                },
            ],
        }
        result = LiteLLMResponsesTransformationHandler._extract_reasoning_input_items_from_thinking_blocks(msg)
        assert len(result) == 1
        assert result[0]["type"] == "reasoning"
        assert result[0]["encrypted_content"] == "opaque_blob_123"
        assert result[0]["id"] == "rs_abc"
        assert result[0]["summary"] == []

    def test_multiple_thinking_blocks(self) -> None:
        msg: Dict[str, Any] = {
            "role": "assistant",
            "content": "Hi",
            "thinking_blocks": [
                {"type": "thinking", "thinking": "step1", "encrypted_content": "enc1", "id": "rs_1"},
                {"type": "thinking", "thinking": "step2", "encrypted_content": "enc2", "id": "rs_2"},
            ],
        }
        result = LiteLLMResponsesTransformationHandler._extract_reasoning_input_items_from_thinking_blocks(msg)
        assert len(result) == 2
        assert result[0]["encrypted_content"] == "enc1"
        assert result[1]["encrypted_content"] == "enc2"

    def test_skips_blocks_without_encrypted_content(self) -> None:
        msg: Dict[str, Any] = {
            "role": "assistant",
            "content": "Hi",
            "thinking_blocks": [
                {"type": "thinking", "thinking": "no encryption here"},
                {"type": "thinking", "thinking": "has it", "encrypted_content": "enc_yes"},
            ],
        }
        result = LiteLLMResponsesTransformationHandler._extract_reasoning_input_items_from_thinking_blocks(msg)
        assert len(result) == 1
        assert result[0]["encrypted_content"] == "enc_yes"

    def test_skips_redacted_thinking_blocks(self) -> None:
        msg: Dict[str, Any] = {
            "role": "assistant",
            "content": "Hi",
            "thinking_blocks": [
                {"type": "redacted_thinking", "data": "base64data"},
                {"type": "thinking", "thinking": "ok", "encrypted_content": "enc"},
            ],
        }
        result = LiteLLMResponsesTransformationHandler._extract_reasoning_input_items_from_thinking_blocks(msg)
        assert len(result) == 1
        assert result[0]["type"] == "reasoning"

    def test_no_thinking_blocks_returns_empty(self) -> None:
        msg: Dict[str, Any] = {"role": "assistant", "content": "Hello"}
        result = LiteLLMResponsesTransformationHandler._extract_reasoning_input_items_from_thinking_blocks(msg)
        assert result == []

    def test_empty_thinking_blocks_returns_empty(self) -> None:
        msg: Dict[str, Any] = {"role": "assistant", "content": "Hello", "thinking_blocks": []}
        result = LiteLLMResponsesTransformationHandler._extract_reasoning_input_items_from_thinking_blocks(msg)
        assert result == []

    def test_none_thinking_blocks_returns_empty(self) -> None:
        msg: Dict[str, Any] = {"role": "assistant", "content": "Hello", "thinking_blocks": None}
        result = LiteLLMResponsesTransformationHandler._extract_reasoning_input_items_from_thinking_blocks(msg)
        assert result == []

    def test_block_without_id_omits_id(self) -> None:
        msg: Dict[str, Any] = {
            "role": "assistant",
            "content": "Hi",
            "thinking_blocks": [
                {"type": "thinking", "thinking": "thought", "encrypted_content": "enc_no_id"},
            ],
        }
        result = LiteLLMResponsesTransformationHandler._extract_reasoning_input_items_from_thinking_blocks(msg)
        assert len(result) == 1
        assert "id" not in result[0]
        assert result[0]["encrypted_content"] == "enc_no_id"


# ---------------------------------------------------------------------------
# convert_chat_completion_messages_to_responses_api
# ---------------------------------------------------------------------------

class TestConvertMessagesWithThinkingBlocks:
    """Tests for the full message conversion with thinking_blocks."""

    def test_assistant_message_with_thinking_blocks_emits_reasoning_before_content(
        self, handler: LiteLLMResponsesTransformationHandler
    ) -> None:
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "Hi there",
                "thinking_blocks": [
                    {"type": "thinking", "thinking": "greet", "encrypted_content": "enc1", "id": "rs_1"},
                ],
            },
            {"role": "user", "content": "How are you?"},
        ]
        input_items, _ = handler.convert_chat_completion_messages_to_responses_api(messages)

        # Should have: user msg, reasoning item, assistant msg, user msg
        assert len(input_items) == 4
        assert input_items[0]["type"] == "message"
        assert input_items[0]["role"] == "user"
        assert input_items[1]["type"] == "reasoning"
        assert input_items[1]["encrypted_content"] == "enc1"
        assert input_items[2]["type"] == "message"
        assert input_items[2]["role"] == "assistant"
        assert input_items[3]["type"] == "message"
        assert input_items[3]["role"] == "user"

    def test_assistant_with_tool_calls_and_thinking_blocks(
        self, handler: LiteLLMResponsesTransformationHandler
    ) -> None:
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                    }
                ],
                "thinking_blocks": [
                    {"type": "thinking", "thinking": "need weather", "encrypted_content": "enc_tc"},
                ],
            },
        ]
        input_items, _ = handler.convert_chat_completion_messages_to_responses_api(messages)

        # reasoning item first, then function_call
        assert input_items[0]["type"] == "reasoning"
        assert input_items[0]["encrypted_content"] == "enc_tc"
        assert input_items[1]["type"] == "function_call"
        assert input_items[1]["name"] == "get_weather"

    def test_no_thinking_blocks_unchanged(
        self, handler: LiteLLMResponsesTransformationHandler
    ) -> None:
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        input_items, _ = handler.convert_chat_completion_messages_to_responses_api(messages)
        assert len(input_items) == 2
        assert all(item["type"] == "message" for item in input_items)


# ---------------------------------------------------------------------------
# _convert_response_output_to_choices (response direction)
# ---------------------------------------------------------------------------

class TestResponseOutputWithEncryptedContent:
    """Tests that encrypted_content from ResponseReasoningItem surfaces as thinking_blocks."""

    def test_reasoning_with_encrypted_content_returns_thinking_blocks(self) -> None:
        """When response contains a reasoning item with encrypted_content,
        the resulting Message should have thinking_blocks."""
        # Use the real OpenAI types for isinstance checks
        from openai.types.responses import ResponseOutputMessage, ResponseReasoningItem

        # Create proper mock objects that pass isinstance checks
        reasoning = MagicMock(spec=ResponseReasoningItem)
        reasoning.summary = [MagicMock(text="I'm thinking...")]
        reasoning.encrypted_content = "encrypted_blob_xyz"
        reasoning.id = "rs_resp_1"

        output = MagicMock(spec=ResponseOutputMessage)
        output.role = "assistant"
        content = MagicMock()
        content.text = "Hello!"
        content.annotations = None
        output.content = [content]

        choices = LiteLLMResponsesTransformationHandler._convert_response_output_to_choices(
            [reasoning, output]
        )

        assert len(choices) == 1
        msg = choices[0].message
        assert msg.content == "Hello!"
        assert msg.reasoning_content == "I'm thinking..."
        assert msg.thinking_blocks is not None
        assert len(msg.thinking_blocks) == 1
        block = msg.thinking_blocks[0]
        assert block["type"] == "thinking"
        assert block["encrypted_content"] == "encrypted_blob_xyz"
        assert block["id"] == "rs_resp_1"
        assert block["thinking"] == "I'm thinking..."

    def test_reasoning_without_encrypted_content_no_thinking_blocks(self) -> None:
        """When reasoning item has no encrypted_content, thinking_blocks should be None."""
        from openai.types.responses import ResponseOutputMessage, ResponseReasoningItem

        reasoning = MagicMock(spec=ResponseReasoningItem)
        reasoning.summary = [MagicMock(text="just a summary")]
        reasoning.encrypted_content = None
        reasoning.id = "rs_no_enc"

        output = MagicMock(spec=ResponseOutputMessage)
        output.role = "assistant"
        content = MagicMock()
        content.text = "Response"
        content.annotations = None
        output.content = [content]

        choices = LiteLLMResponsesTransformationHandler._convert_response_output_to_choices(
            [reasoning, output]
        )

        assert len(choices) == 1
        msg = choices[0].message
        # thinking_blocks should not be set (None or absent)
        assert getattr(msg, "thinking_blocks", None) is None

    def test_reasoning_with_encrypted_content_and_tool_calls(self) -> None:
        """When reasoning with encrypted_content precedes tool calls,
        thinking_blocks should appear on the tool_calls message."""
        from openai.types.responses import (
            ResponseFunctionToolCall,
            ResponseReasoningItem,
        )

        reasoning = MagicMock(spec=ResponseReasoningItem)
        reasoning.summary = [MagicMock(text="I need to call a tool")]
        reasoning.encrypted_content = "enc_tool"
        reasoning.id = "rs_tool_1"

        tool_call = MagicMock(spec=ResponseFunctionToolCall)
        tool_call.call_id = "call_456"
        tool_call.name = "search"
        tool_call.arguments = '{"q": "test"}'
        tool_call.type = "function_call"

        choices = LiteLLMResponsesTransformationHandler._convert_response_output_to_choices(
            [reasoning, tool_call]
        )

        assert len(choices) == 1
        msg = choices[0].message
        assert msg.tool_calls is not None
        assert msg.thinking_blocks is not None
        assert msg.thinking_blocks[0]["encrypted_content"] == "enc_tool"


# ---------------------------------------------------------------------------
# transform_request auto-include injection
# ---------------------------------------------------------------------------

class TestTransformRequestAutoInclude:
    """Tests that transform_request auto-injects include: ["reasoning.encrypted_content"]."""

    @staticmethod
    def _make_logging_obj() -> MagicMock:
        return MagicMock()

    def test_auto_includes_reasoning_encrypted_content(
        self, handler: LiteLLMResponsesTransformationHandler
    ) -> None:
        messages = [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": "Hello",
                "thinking_blocks": [
                    {"type": "thinking", "thinking": "greet", "encrypted_content": "enc_auto"},
                ],
            },
            {"role": "user", "content": "Follow-up"},
        ]
        result = handler.transform_request(
            model="openai/o4-mini",
            messages=messages,
            optional_params={},
            litellm_params={"api_key": "test", "api_base": None},
            headers={},
            litellm_logging_obj=self._make_logging_obj(),
        )
        # The include parameter should be auto-injected
        assert "reasoning.encrypted_content" in (result.get("include") or [])

    def test_no_auto_include_without_encrypted_content(
        self, handler: LiteLLMResponsesTransformationHandler
    ) -> None:
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Follow-up"},
        ]
        result = handler.transform_request(
            model="openai/o4-mini",
            messages=messages,
            optional_params={},
            litellm_params={"api_key": "test", "api_base": None},
            headers={},
            litellm_logging_obj=self._make_logging_obj(),
        )
        # No include should be set (or if set, shouldn't have our value)
        include = result.get("include") or []
        assert "reasoning.encrypted_content" not in include

    def test_auto_include_does_not_duplicate(
        self, handler: LiteLLMResponsesTransformationHandler
    ) -> None:
        messages = [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": "Hello",
                "thinking_blocks": [
                    {"type": "thinking", "thinking": "greet", "encrypted_content": "enc"},
                ],
            },
            {"role": "user", "content": "Follow-up"},
        ]
        result = handler.transform_request(
            model="openai/o4-mini",
            messages=messages,
            optional_params={"include": ["reasoning.encrypted_content"]},
            litellm_params={"api_key": "test", "api_base": None},
            headers={},
            litellm_logging_obj=self._make_logging_obj(),
        )
        include = result.get("include") or []
        # Should appear exactly once
        assert include.count("reasoning.encrypted_content") == 1

    def test_auto_include_preserves_existing_includes(
        self, handler: LiteLLMResponsesTransformationHandler
    ) -> None:
        messages = [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": "Hello",
                "thinking_blocks": [
                    {"type": "thinking", "thinking": "greet", "encrypted_content": "enc"},
                ],
            },
            {"role": "user", "content": "Follow-up"},
        ]
        result = handler.transform_request(
            model="openai/o4-mini",
            messages=messages,
            optional_params={"include": ["file_search_call.results"]},
            litellm_params={"api_key": "test", "api_base": None},
            headers={},
            litellm_logging_obj=self._make_logging_obj(),
        )
        include = result.get("include") or []
        assert "file_search_call.results" in include
        assert "reasoning.encrypted_content" in include


# ---------------------------------------------------------------------------
# Round-trip test
# ---------------------------------------------------------------------------

class TestRoundTrip:
    """End-to-end: thinking_blocks in → reasoning items → response with encrypted → thinking_blocks out."""

    def test_round_trip_preserves_encrypted_content(
        self, handler: LiteLLMResponsesTransformationHandler
    ) -> None:
        """Verify the full round-trip: input thinking_blocks produce reasoning items,
        and a response with encrypted_content produces thinking_blocks back."""
        # 1. Input direction
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "First response",
                "thinking_blocks": [
                    {
                        "type": "thinking",
                        "thinking": "Initial reasoning",
                        "encrypted_content": "round_trip_enc",
                        "id": "rs_rt1",
                    },
                ],
            },
            {"role": "user", "content": "Follow-up"},
        ]
        input_items, _ = handler.convert_chat_completion_messages_to_responses_api(messages)

        # Verify reasoning items were created
        reasoning_items = [i for i in input_items if i.get("type") == "reasoning"]
        assert len(reasoning_items) == 1
        assert reasoning_items[0]["encrypted_content"] == "round_trip_enc"

        # 2. Response direction
        from openai.types.responses import ResponseOutputMessage, ResponseReasoningItem

        resp_reasoning = MagicMock(spec=ResponseReasoningItem)
        resp_reasoning.summary = [MagicMock(text="New reasoning")]
        resp_reasoning.encrypted_content = "new_enc_blob"
        resp_reasoning.id = "rs_rt2"

        resp_output = MagicMock(spec=ResponseOutputMessage)
        resp_output.role = "assistant"
        resp_output.id = "msg_rt2"
        content = MagicMock()
        content.text = "New response"
        content.annotations = None
        resp_output.content = [content]

        choices = LiteLLMResponsesTransformationHandler._convert_response_output_to_choices(
            [resp_reasoning, resp_output]
        )

        msg = choices[0].message
        assert msg.thinking_blocks is not None
        assert msg.thinking_blocks[0]["encrypted_content"] == "new_enc_blob"
        assert msg.thinking_blocks[0]["id"] == "rs_rt2"
        assert msg.provider_specific_fields is not None
        assert msg.provider_specific_fields["_response_message_id"] == "msg_rt2"
        assert msg.reasoning_content == "New reasoning"


class TestReasoningItemSummaryField:
    """Tests for the required 'summary' field on reasoning input items."""

    def test_reasoning_items_include_summary_field(self) -> None:
        """OpenAI requires 'summary: []' on reasoning input items."""
        msg: Dict[str, Any] = {
            "role": "assistant",
            "content": "42",
            "thinking_blocks": [
                {"type": "thinking", "thinking": "", "encrypted_content": "enc", "id": "rs_1"},
            ],
        }
        result = LiteLLMResponsesTransformationHandler._extract_reasoning_input_items_from_thinking_blocks(msg)
        assert result[0]["summary"] == []

    def test_response_reasoning_item_with_summary_none(self) -> None:
        """ResponseReasoningItem with summary=None must not raise TypeError."""
        from openai.types.responses import ResponseOutputMessage, ResponseReasoningItem

        reasoning = MagicMock(spec=ResponseReasoningItem)
        reasoning.summary = None
        reasoning.encrypted_content = None
        reasoning.id = "rs_null"

        output_msg = MagicMock(spec=ResponseOutputMessage)
        output_msg.role = "assistant"
        output_msg.id = "msg_null"
        content = MagicMock()
        content.text = "answer"
        content.annotations = None
        output_msg.content = [content]

        choices = LiteLLMResponsesTransformationHandler._convert_response_output_to_choices(
            [reasoning, output_msg]
        )
        assert len(choices) == 1
        # reasoning_content should be empty since summary was None
        assert choices[0].message.reasoning_content == ""


class TestAssistantMessageStatusAndId:
    """Tests for status and id on assistant messages in responses API input."""

    @pytest.fixture()
    def handler(self) -> LiteLLMResponsesTransformationHandler:
        return LiteLLMResponsesTransformationHandler()

    def test_assistant_content_none_normalized_to_empty_string(
        self, handler: LiteLLMResponsesTransformationHandler
    ) -> None:
        """Explicit content=None and missing content key produce same output for assistant."""
        msg_missing = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "thinking_blocks": [
                {"type": "thinking", "thinking": "", "encrypted_content": "enc", "id": "rs_1"},
            ]},
        ]
        msg_explicit_none = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": None, "thinking_blocks": [
                {"type": "thinking", "thinking": "", "encrypted_content": "enc", "id": "rs_1"},
            ]},
        ]
        items_missing, _ = handler.convert_chat_completion_messages_to_responses_api(msg_missing)
        items_none, _ = handler.convert_chat_completion_messages_to_responses_api(msg_explicit_none)
        assert items_missing == items_none

    def test_thinking_blocks_preserved_when_content_none_no_tool_calls(
        self, handler: LiteLLMResponsesTransformationHandler
    ) -> None:
        """thinking_blocks must not be silently dropped when content=None and no tool_calls."""
        messages = [
            {"role": "user", "content": "Think about this"},
            {
                "role": "assistant",
                "content": None,
                "thinking_blocks": [
                    {
                        "type": "thinking",
                        "thinking": "deep thought",
                        "encrypted_content": "enc_data",
                        "id": "rs_think_1",
                    },
                ],
            },
        ]
        items, _ = handler.convert_chat_completion_messages_to_responses_api(messages)
        reasoning_items = [i for i in items if i.get("type") == "reasoning"]
        assert len(reasoning_items) == 1, (
            "thinking_blocks should produce a reasoning item even when content is None"
        )
        assert reasoning_items[0]["id"] == "rs_think_1"
        encrypted = reasoning_items[0].get("encrypted_content")
        assert encrypted == "enc_data"

    def test_assistant_message_has_status_completed(
        self, handler: LiteLLMResponsesTransformationHandler
    ) -> None:
        """Assistant messages must include status: 'completed' for Responses API."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        items, _ = handler.convert_chat_completion_messages_to_responses_api(messages)
        assistant_items = [i for i in items if i.get("role") == "assistant"]
        assert len(assistant_items) == 1
        assert assistant_items[0]["status"] == "completed"

    def test_user_message_has_no_status(
        self, handler: LiteLLMResponsesTransformationHandler
    ) -> None:
        """User messages must NOT include a status field."""
        messages = [{"role": "user", "content": "Hi"}]
        items, _ = handler.convert_chat_completion_messages_to_responses_api(messages)
        assert "status" not in items[0]

    def test_assistant_message_gets_id_from_thinking_blocks(
        self, handler: LiteLLMResponsesTransformationHandler
    ) -> None:
        """When provider_specific_fields carry _response_message_id, it goes on the assistant message."""
        messages = [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": "Hello",
                "thinking_blocks": [
                    {
                        "type": "thinking",
                        "thinking": "",
                        "encrypted_content": "enc",
                        "id": "rs_1",
                    },
                ],
                "provider_specific_fields": {
                    "_response_message_id": "msg_42",
                },
            },
            {"role": "user", "content": "Follow-up"},
        ]
        items, _ = handler.convert_chat_completion_messages_to_responses_api(messages)
        assistant_items = [i for i in items if i.get("role") == "assistant"]
        assert assistant_items[0]["id"] == "msg_42"

    def test_assistant_message_no_id_without_thinking_blocks(
        self, handler: LiteLLMResponsesTransformationHandler
    ) -> None:
        """Without thinking_blocks, assistant message should not have an id field."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        items, _ = handler.convert_chat_completion_messages_to_responses_api(messages)
        assistant_items = [i for i in items if i.get("role") == "assistant"]
        assert "id" not in assistant_items[0]


class TestResponseMessageIdTagging:
    """Tests for _response_message_id in provider_specific_fields from response output."""

    def test_thinking_blocks_tagged_with_message_id(self) -> None:
        """Response should store _response_message_id in provider_specific_fields."""
        from openai.types.responses import ResponseOutputMessage, ResponseReasoningItem

        reasoning = MagicMock(spec=ResponseReasoningItem)
        reasoning.summary = []
        reasoning.encrypted_content = "enc_abc"
        reasoning.id = "rs_x1"

        output_msg = MagicMock(spec=ResponseOutputMessage)
        output_msg.role = "assistant"
        output_msg.id = "msg_x1"
        content = MagicMock()
        content.text = "answer"
        content.annotations = None
        output_msg.content = [content]

        choices = LiteLLMResponsesTransformationHandler._convert_response_output_to_choices(
            [reasoning, output_msg]
        )
        tb = choices[0].message.thinking_blocks
        assert tb is not None
        assert "_response_message_id" not in tb[0]
        psf = choices[0].message.provider_specific_fields
        assert psf is not None
        assert psf["_response_message_id"] == "msg_x1"

    def test_thinking_blocks_no_tag_without_message_id(self) -> None:
        """If output message has no id, no _response_message_id should be set."""
        from openai.types.responses import ResponseOutputMessage, ResponseReasoningItem

        reasoning = MagicMock(spec=ResponseReasoningItem)
        reasoning.summary = []
        reasoning.encrypted_content = "enc_abc"
        reasoning.id = "rs_x2"

        output_msg = MagicMock(spec=ResponseOutputMessage)
        output_msg.role = "assistant"
        output_msg.id = None
        content = MagicMock()
        content.text = "answer"
        content.annotations = None
        output_msg.content = [content]

        choices = LiteLLMResponsesTransformationHandler._convert_response_output_to_choices(
            [reasoning, output_msg]
        )
        tb = choices[0].message.thinking_blocks
        assert tb is not None
        assert "_response_message_id" not in tb[0]
        psf = choices[0].message.provider_specific_fields
        assert psf is None
