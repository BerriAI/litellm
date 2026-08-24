"""
Unit tests for Anthropic Messages Guardrail Translation Handler

Tests the handler's ability to process streaming output for Anthropic Messages API
with guardrail transformations, specifically testing edge cases with empty choices.
"""

import json
from typing import Any, Literal, Optional
from unittest.mock import MagicMock, patch

import pytest


from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms.anthropic.chat.guardrail_translation.handler import (
    AnthropicMessagesHandler,
)
from litellm.types.utils import GenericGuardrailAPIInputs


class MockPassThroughGuardrail(CustomGuardrail):
    """Mock guardrail that passes through without blocking - for testing streaming fallback behavior"""

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        """Simply return inputs unchanged"""
        return inputs


class MockDynamicGuardrail(CustomGuardrail):
    """Mock guardrail that records dynamic params from request metadata."""

    def __init__(self, guardrail_name: str):
        super().__init__(guardrail_name=guardrail_name)
        self.dynamic_params: Optional[dict] = None

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        self.dynamic_params = self.get_guardrail_dynamic_request_body_params(
            request_data
        )
        return inputs


class MockRecordingGuardrail(CustomGuardrail):
    """Mock guardrail that records the request_data it was handed."""

    def __init__(self, guardrail_name: str):
        super().__init__(guardrail_name=guardrail_name)
        self.request_data: Optional[dict] = None

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        self.request_data = request_data
        return inputs


class MockMaskingGuardrail(CustomGuardrail):
    """Capture request inputs and mask one known prohibited value."""

    def __init__(self, skip_system_message_in_guardrail: Optional[bool] = True):
        super().__init__(guardrail_name="masking-test")
        self.skip_system_message_in_guardrail = skip_system_message_in_guardrail
        self.inputs: Optional[GenericGuardrailAPIInputs] = None

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        self.inputs = inputs.copy()
        masked_inputs = inputs.copy()
        masked_inputs["texts"] = [
            "[MASKED]" if text == "prohibited correction" else text for text in inputs.get("texts", [])
        ]
        return masked_inputs


class MockCompactingGuardrail(CustomGuardrail):
    """Stand in for a compaction guardrail that rewrites `structured_messages` wholesale."""

    def __init__(self, replacement_messages: list):
        super().__init__(guardrail_name="compacting-test")
        self.replacement_messages = replacement_messages
        self.inputs: Optional[GenericGuardrailAPIInputs] = None

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        self.inputs = inputs.copy()
        rewritten = inputs.copy()
        # A new list object -- this is what signals a rewrite to the handler.
        rewritten["structured_messages"] = list(self.replacement_messages)
        return rewritten


class MockStructuredMaskingGuardrail(CustomGuardrail):
    """Mask an email in texts and in a rebuilt structured view, like a PII-masking guardrail (LIT-5696)."""

    def __init__(self):
        super().__init__(guardrail_name="structured-masking-test")

    @staticmethod
    def _mask(text: str) -> str:
        return text.replace("bob@example.com", "<EMAIL>")

    def _mask_content(self, content: object) -> object:
        if isinstance(content, str):
            return self._mask(content)
        if not isinstance(content, list):
            return content
        return [
            {**block, "text": self._mask(block["text"])}
            if isinstance(block, dict) and isinstance(block.get("text"), str)
            else block
            for block in content
        ]

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        masked = inputs.copy()
        masked["texts"] = [self._mask(text) for text in inputs.get("texts", [])]
        structured = inputs.get("structured_messages")
        if structured is not None:
            masked["structured_messages"] = [
                {**message, "content": self._mask_content(message.get("content"))} for message in structured
            ]
        return masked


class TestAnthropicMessagesHandlerStreamingRequestData:
    """Post-call guardrails on streaming /v1/messages receive the response and identity metadata"""

    @pytest.mark.asyncio
    async def test_terminal_chunk_passes_assembled_response_and_metadata(self):
        from litellm.proxy._types import UserAPIKeyAuth
        from litellm.types.utils import Choices, Message, ModelResponse

        handler = AnthropicMessagesHandler()
        guardrail = MockRecordingGuardrail(guardrail_name="test")
        mock_response = ModelResponse(
            id="msg_123",
            created=1234567890,
            model="claude-sonnet-4-5",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(content="Hello world", role="assistant"),
                )
            ],
        )

        with (
            patch.object(handler, "_check_streaming_has_ended", return_value=True),
            patch(
                "litellm.llms.anthropic.chat.guardrail_translation.handler.AnthropicPassthroughLoggingHandler._build_complete_streaming_response",
                return_value=mock_response,
            ),
        ):
            await handler.process_output_streaming_response(
                responses_so_far=[b"data: some chunk"],
                guardrail_to_apply=guardrail,
                litellm_logging_obj=MagicMock(),
                user_api_key_dict=UserAPIKeyAuth(user_id="u-1", team_id="t-1"),
                request_data={"model": "claude-sonnet-4-5"},
            )

        assert guardrail.request_data is not None
        assert guardrail.request_data["response"] is mock_response
        assert (
            guardrail.request_data["litellm_metadata"]["user_api_key_user_id"] == "u-1"
        )

    @pytest.mark.asyncio
    async def test_mid_stream_chunk_passes_responses_so_far_and_metadata(self):
        from litellm.proxy._types import UserAPIKeyAuth

        handler = AnthropicMessagesHandler()
        guardrail = MockRecordingGuardrail(guardrail_name="test")
        responses_so_far = [b"data: some chunk"]

        with (
            patch.object(handler, "_check_streaming_has_ended", return_value=False),
            patch.object(
                handler, "get_streaming_string_so_far", return_value="partial text"
            ),
        ):
            await handler.process_output_streaming_response(
                responses_so_far=responses_so_far,
                guardrail_to_apply=guardrail,
                litellm_logging_obj=MagicMock(),
                user_api_key_dict=UserAPIKeyAuth(user_id="u-1", team_id="t-1"),
                request_data={"model": "claude-sonnet-4-5"},
            )

        assert guardrail.request_data is not None
        assert guardrail.request_data["responses"] is responses_so_far
        assert (
            guardrail.request_data["litellm_metadata"]["user_api_key_user_id"] == "u-1"
        )


class TestAnthropicMessagesHandlerStreamingOutputProcessing:
    """Test streaming output processing functionality"""

    @pytest.mark.asyncio
    async def test_process_output_streaming_response_empty_model_response(self):
        """Test that streaming response with None model_response doesn't raise error

        This test verifies the fix for the bug where accessing model_response.choices[0]
        would raise an error when _build_complete_streaming_response returns None.
        """
        handler = AnthropicMessagesHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        # Mock _check_streaming_has_ended to return True (stream ended)
        # and _build_complete_streaming_response to return None
        with (
            patch.object(handler, "_check_streaming_has_ended", return_value=True),
            patch(
                "litellm.llms.anthropic.chat.guardrail_translation.handler.AnthropicPassthroughLoggingHandler._build_complete_streaming_response",
                return_value=None,
            ),
        ):
            responses_so_far = [b"data: some chunk"]

            # This should not raise an error
            result = await handler.process_output_streaming_response(
                responses_so_far=responses_so_far,
                guardrail_to_apply=guardrail,
                litellm_logging_obj=MagicMock(),
            )

            # Should return the responses unchanged
            assert result == responses_so_far


class TestAnthropicMessagesHandlerInputProcessing:
    """Test input processing preserves litellm_metadata for dynamic guardrails."""

    @pytest.mark.asyncio
    async def test_process_input_messages_preserves_litellm_metadata_guardrails(self):
        handler = AnthropicMessagesHandler()
        guardrail = MockDynamicGuardrail(guardrail_name="cygnal-monitor")

        data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hello"}],
            "litellm_metadata": {
                "guardrails": [
                    {"cygnal-monitor": {"extra_body": {"policy_id": "policy-123"}}}
                ]
            },
        }

        with patch("litellm.proxy.proxy_server.premium_user", True):
            await handler.process_input_messages(
                data=data, guardrail_to_apply=guardrail
            )

        assert data.get("litellm_metadata", {}).get("guardrails")
        assert guardrail.dynamic_params == {"policy_id": "policy-123"}

    @pytest.mark.asyncio
    async def test_midturn_system_correction_is_guardrailed_when_top_level_system_is_skipped(
        self,
    ):
        handler = AnthropicMessagesHandler()
        guardrail = MockMaskingGuardrail()
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "system": "trusted top-level system prompt",
            "messages": [
                {"role": "user", "content": "safe text"},
                {
                    "role": "system",
                    "content": [
                        {"type": "unsupported", "text": "discarded text"},
                        {"type": "text", "text": "prohibited correction"},
                    ],
                },
            ],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.inputs is not None
        assert guardrail.inputs["texts"] == ["safe text", "prohibited correction"]
        assert "trusted top-level system prompt" not in guardrail.inputs["texts"]
        assert data["messages"][1]["content"][0]["text"] == "discarded text"
        assert data["messages"][1]["content"][1]["text"] == "[MASKED]"

    @pytest.mark.asyncio
    async def test_string_midturn_system_correction_is_guardrailed(self):
        handler = AnthropicMessagesHandler()
        guardrail = MockMaskingGuardrail()
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "system", "content": "prohibited correction"}],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.inputs is not None
        assert guardrail.inputs["texts"] == ["prohibited correction"]
        assert data["messages"][0]["content"] == "[MASKED]"

    @pytest.mark.asyncio
    async def test_unsupported_midturn_system_content_is_not_guardrailed(self):
        handler = AnthropicMessagesHandler()
        guardrail = MockMaskingGuardrail()
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "image", "source": {"type": "url"}}],
                }
            ],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.inputs is None

    @pytest.mark.asyncio
    async def test_skip_system_message_excludes_only_hoisted_top_level_system(self):
        handler = AnthropicMessagesHandler()
        guardrail = MockMaskingGuardrail()
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "system": "trusted top-level system prompt",
            "messages": [
                {"role": "user", "content": "safe text"},
                {"role": "system", "content": "prohibited correction"},
                {"role": "user", "content": "continue"},
            ],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.inputs is not None
        structured = guardrail.inputs["structured_messages"]
        assert [m["role"] for m in structured] == ["user", "system", "user"]
        assert structured[1]["content"] == "prohibited correction"

    @pytest.mark.asyncio
    async def test_default_skip_false_scans_midturn_system_and_hoists_top_level_system(
        self,
    ):
        handler = AnthropicMessagesHandler()
        guardrail = MockMaskingGuardrail(skip_system_message_in_guardrail=None)
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "system": "trusted top-level system prompt",
            "messages": [
                {"role": "user", "content": "safe text"},
                {"role": "system", "content": "prohibited correction"},
            ],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.inputs is not None
        assert guardrail.inputs["texts"] == ["safe text", "prohibited correction"]
        structured = guardrail.inputs["structured_messages"]
        assert [m["role"] for m in structured] == ["system", "user", "system"]
        assert structured[0]["content"] == "trusted top-level system prompt"
        assert data["messages"][1]["content"] == "[MASKED]"

    @pytest.mark.asyncio
    async def test_bedrock_masking_slice_is_unavailable_when_top_level_system_is_included(
        self,
    ):
        from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
            BedrockGuardrail,
        )

        handler = AnthropicMessagesHandler()
        guardrail = MockMaskingGuardrail(skip_system_message_in_guardrail=None)
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "system": "trusted top-level system prompt",
            "messages": [
                {"role": "user", "content": "safe text"},
                {"role": "system", "content": "prohibited correction"},
                {"role": "user", "content": "latest question"},
            ],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.inputs is not None
        texts = guardrail.inputs["texts"]
        structured = guardrail.inputs["structured_messages"]

        bedrock = BedrockGuardrail(guardrailIdentifier="gi", guardrailVersion="1")
        assert sum(bedrock._count_message_texts(m) for m in structured) == len(texts) + 1
        latest_user_index = bedrock._find_latest_message_index(structured, target_role="user")
        assert (
            bedrock._locate_message_texts_slice(
                structured_messages=structured,
                target_index=latest_user_index,
                texts=texts,
            )
            is None
        )
        assert (
            bedrock._merge_masked_texts(
                masked_texts=["{MASKED}"],
                texts=texts,
                scanned_slice=None,
                scanned_role_subset=True,
            )
            == texts
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize("skip_system_message_in_guardrail", [True, None])
    async def test_midturn_system_text_extraction_matches_translation_in_both_skip_modes(
        self,
        skip_system_message_in_guardrail: Optional[bool],
    ):
        from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
            BedrockGuardrail,
        )

        handler = AnthropicMessagesHandler()
        guardrail = MockMaskingGuardrail(skip_system_message_in_guardrail=skip_system_message_in_guardrail)
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": "safe text"},
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": ""},
                        {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}},
                        {"type": "text", "text": "prohibited correction"},
                    ],
                },
            ],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.inputs is not None
        texts = guardrail.inputs["texts"]
        structured = guardrail.inputs["structured_messages"]
        assert texts == ["safe text", "prohibited correction"]
        bedrock = BedrockGuardrail(guardrailIdentifier="gi", guardrailVersion="1")
        assert sum(bedrock._count_message_texts(m) for m in structured) == len(texts)
        assert data["messages"][1]["content"][2]["text"] == "[MASKED]"

    @pytest.mark.asyncio
    async def test_bedrock_masking_slice_stays_aligned_with_midturn_system(self):
        from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
            BedrockGuardrail,
        )

        handler = AnthropicMessagesHandler()
        guardrail = MockMaskingGuardrail()
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "system": "trusted top-level system prompt",
            "messages": [
                {"role": "user", "content": "safe text"},
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "prohibited correction"},
                        {"type": "text", "text": "second correction"},
                    ],
                },
                {"role": "user", "content": "latest question"},
            ],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.inputs is not None
        texts = guardrail.inputs["texts"]
        structured = guardrail.inputs["structured_messages"]

        bedrock = BedrockGuardrail(guardrailIdentifier="gi", guardrailVersion="1")
        total = sum(bedrock._count_message_texts(m) for m in structured)
        assert total == len(texts)

        latest_user_index = bedrock._find_latest_message_index(structured, target_role="user")
        assert latest_user_index == 2
        scanned_slice = bedrock._locate_message_texts_slice(
            structured_messages=structured,
            target_index=latest_user_index,
            texts=texts,
        )
        assert scanned_slice == (3, 1)

        merged = bedrock._merge_masked_texts(
            masked_texts=["{MASKED}"],
            texts=texts,
            scanned_slice=scanned_slice,
            scanned_role_subset=True,
        )
        assert merged == [
            "safe text",
            "prohibited correction",
            "second correction",
            "{MASKED}",
        ]

    @pytest.mark.asyncio
    async def test_compaction_rewrite_keeps_midturn_system_messages(self):
        handler = AnthropicMessagesHandler()
        guardrail = MockCompactingGuardrail(
            replacement_messages=[
                {"role": "user", "content": "compacted history"},
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "use the corrected result"}],
                },
                {"role": "user", "content": "continue"},
            ]
        )
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "system": "trusted top-level system prompt",
            "messages": [
                {"role": "user", "content": "original history"},
                {"role": "system", "content": "use the corrected result"},
                {"role": "user", "content": "continue"},
            ],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert [m["role"] for m in data["messages"]] == ["user", "system", "user"]
        assert data["messages"][1]["content"] == [{"type": "text", "text": "use the corrected result"}]
        assert data["messages"][0]["content"] == [{"type": "text", "text": "compacted history"}]
        assert data["messages"][2]["content"] == [{"type": "text", "text": "continue"}]
        assert data["system"] == "trusted top-level system prompt"

    @pytest.mark.asyncio
    async def test_midturn_system_inside_tool_exchange_keeps_the_pair_intact(self):
        """A system row between an assistant tool call and its result must not split the
        exchange into orphaned halves; it is emitted right after the exchange instead."""
        handler = AnthropicMessagesHandler()
        guardrail = MockCompactingGuardrail(
            replacement_messages=[
                {"role": "user", "content": "run the tool"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "system", "content": "use the corrected result"},
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            ]
        )
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": "run the tool"},
                {"role": "system", "content": "use the corrected result"},
            ],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert [m["role"] for m in data["messages"]] == ["user", "assistant", "user", "system"]
        assistant_blocks = data["messages"][1]["content"]
        assert any(block.get("type") == "tool_use" and block.get("id") == "call_1" for block in assistant_blocks)
        result_blocks = data["messages"][2]["content"]
        assert [block["type"] for block in result_blocks] == ["tool_result"]
        assert result_blocks[0]["tool_use_id"] == "call_1"
        assert data["messages"][3]["content"] == "use the corrected result"

    @pytest.mark.asyncio
    async def test_compaction_rewrite_does_not_duplicate_hoisted_top_level_system(self):
        handler = AnthropicMessagesHandler()
        guardrail = MockCompactingGuardrail(
            replacement_messages=[
                {"role": "system", "content": "trusted top-level system prompt"},
                {"role": "user", "content": "compacted history"},
                {"role": "system", "content": "use the corrected result"},
            ]
        )
        guardrail.skip_system_message_in_guardrail = None
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "system": "trusted top-level system prompt",
            "messages": [
                {"role": "user", "content": "original history"},
                {"role": "system", "content": "use the corrected result"},
            ],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert [m["role"] for m in data["messages"]] == ["user", "system"]
        assert data["messages"][1]["content"] == "use the corrected result"
        assert data["system"] == "trusted top-level system prompt"

    @pytest.mark.asyncio
    async def test_leading_system_row_appends_to_skipped_top_level_system(
        self,
    ):
        handler = AnthropicMessagesHandler()
        guardrail = MockCompactingGuardrail(
            replacement_messages=[
                {"role": "system", "content": "use the corrected result"},
                {"role": "user", "content": "compacted history"},
            ]
        )
        guardrail.skip_system_message_in_guardrail = True
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "system": "trusted top-level system prompt",
            "messages": [
                {"role": "system", "content": "use the corrected result"},
                {"role": "user", "content": "original history"},
            ],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert [m["role"] for m in data["messages"]] == ["user"]
        assert data["system"] == [
            {"type": "text", "text": "trusted top-level system prompt"},
            {"type": "text", "text": "use the corrected result"},
        ]

    @pytest.mark.asyncio
    async def test_leading_correction_appends_when_top_level_system_hoists_nothing(
        self,
    ):
        handler = AnthropicMessagesHandler()
        guardrail = MockCompactingGuardrail(
            replacement_messages=[
                {"role": "system", "content": "use the corrected result"},
                {"role": "user", "content": "compacted history"},
            ]
        )
        guardrail.skip_system_message_in_guardrail = None
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "system": [{"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}],
            "messages": [
                {"role": "system", "content": "use the corrected result"},
                {"role": "user", "content": "original history"},
            ],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert [m["role"] for m in data["messages"]] == ["user"]
        assert data["system"] == [
            {"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}},
            {"type": "text", "text": "use the corrected result"},
        ]

    @pytest.mark.asyncio
    async def test_leading_correction_replaces_top_level_system_when_hoisted_prompt_is_dropped(
        self,
    ):
        handler = AnthropicMessagesHandler()
        guardrail = MockCompactingGuardrail(
            replacement_messages=[
                {"role": "system", "content": "CLIENT CORRECTION"},
                {"role": "user", "content": "compacted history"},
            ]
        )
        guardrail.skip_system_message_in_guardrail = None
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "system": "TRUSTED",
            "messages": [
                {"role": "system", "content": "CLIENT CORRECTION"},
                {"role": "user", "content": "original history"},
            ],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.inputs is not None
        assert guardrail.inputs["structured_messages"][0] == {
            "role": "system",
            "content": "TRUSTED",
        }
        assert [m["role"] for m in data["messages"]] == ["user"]
        assert data["system"] == [{"type": "text", "text": "CLIENT CORRECTION"}]

    @pytest.mark.asyncio
    async def test_masked_hoisted_system_folds_into_top_level_system(self):
        """LIT-5696: a guardrail-modified top-level prompt must go back through the system
        param; emitting it as messages[0] is rejected by Anthropic, dropping it leaks the
        unmasked original."""
        handler = AnthropicMessagesHandler()
        guardrail = MockStructuredMaskingGuardrail()
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "system": [{"type": "text", "text": "You are helpful. The admin is bob@example.com."}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert data["system"] == [{"type": "text", "text": "You are helpful. The admin is <EMAIL>."}]
        assert [m["role"] for m in data["messages"]] == ["user"]

    @pytest.mark.asyncio
    async def test_client_leading_system_row_folds_into_top_level_system(self):
        """LIT-5696: a client-sent leading system row folds into the system param instead of
        being sent back as messages[0], which Anthropic rejects."""
        handler = AnthropicMessagesHandler()
        guardrail = MockStructuredMaskingGuardrail()
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "You are helpful."}]},
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            ],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert data["system"] == [{"type": "text", "text": "You are helpful."}]
        assert [m["role"] for m in data["messages"]] == ["user"]

    @pytest.mark.asyncio
    async def test_masked_midturn_system_after_user_stays_in_messages(self):
        handler = AnthropicMessagesHandler()
        guardrail = MockStructuredMaskingGuardrail()
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "system": [{"type": "text", "text": "You are helpful. The admin is bob@example.com."}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
                {"role": "system", "content": [{"type": "text", "text": "Mid-turn: admin bob@example.com"}]},
                {"role": "user", "content": [{"type": "text", "text": "next"}]},
            ],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert data["system"] == [{"type": "text", "text": "You are helpful. The admin is <EMAIL>."}]
        assert [m["role"] for m in data["messages"]] == ["user", "assistant", "system", "user"]
        assert data["messages"][2]["content"] == [{"type": "text", "text": "Mid-turn: admin <EMAIL>"}]

    @pytest.mark.asyncio
    async def test_unmodified_structured_copy_leaves_top_level_system_untouched(self):
        handler = AnthropicMessagesHandler()
        guardrail = MockStructuredMaskingGuardrail()
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert data["system"] == "You are helpful."
        assert [m["role"] for m in data["messages"]] == ["user"]

    @pytest.mark.asyncio
    async def test_compaction_rewrite_drops_hoisted_prompt_matched_by_content_copy(self):
        import json

        handler = AnthropicMessagesHandler()
        guardrail = MockCompactingGuardrail(
            replacement_messages=[
                json.loads(json.dumps({"role": "system", "content": "TRUSTED"})),
                {"role": "user", "content": "compacted history"},
                {"role": "system", "content": "CLIENT CORRECTION"},
            ]
        )
        guardrail.skip_system_message_in_guardrail = None
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "system": "TRUSTED",
            "messages": [
                {"role": "user", "content": "original history"},
                {"role": "system", "content": "CLIENT CORRECTION"},
            ],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert [m["role"] for m in data["messages"]] == ["user", "system"]
        assert data["messages"][1]["content"] == "CLIENT CORRECTION"
        assert data["system"] == "TRUSTED"

    @pytest.mark.asyncio
    async def test_compaction_rewrite_preserves_cache_control_on_system_blocks(self):
        """
        `cache_control` on an in-sequence system text block survives the write-back, and is
        copied rather than aliased into the guardrail's own returned list.
        """
        handler = AnthropicMessagesHandler()
        source_cache_control = {"type": "ephemeral"}
        guardrail = MockCompactingGuardrail(
            replacement_messages=[
                {"role": "user", "content": "compacted history"},
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "use the corrected result",
                            "cache_control": source_cache_control,
                        }
                    ],
                },
            ]
        )
        guardrail.skip_system_message_in_guardrail = True
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": "original history"},
                {"role": "system", "content": "use the corrected result"},
            ],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert data["messages"][1]["content"] == [
            {
                "type": "text",
                "text": "use the corrected result",
                "cache_control": {"type": "ephemeral"},
            }
        ]
        assert data["messages"][1]["content"][0]["cache_control"] is not source_cache_control

    @pytest.mark.asyncio
    async def test_compaction_rewrite_rstrips_trailing_assistant_in_each_run(self):
        handler = AnthropicMessagesHandler()
        guardrail = MockCompactingGuardrail(
            replacement_messages=[
                {"role": "user", "content": "compacted history"},
                {"role": "assistant", "content": "earlier  "},
                {"role": "system", "content": "use the corrected result"},
                {"role": "user", "content": "continue"},
                {"role": "assistant", "content": "prefill  "},
            ]
        )
        guardrail.skip_system_message_in_guardrail = True
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": "original history"},
                {"role": "system", "content": "use the corrected result"},
            ],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert [m["role"] for m in data["messages"]] == [
            "user",
            "assistant",
            "system",
            "user",
            "assistant",
        ]
        assert data["messages"][1]["content"] == [{"type": "text", "text": "earlier"}]
        assert data["messages"][-1]["content"] == [{"type": "text", "text": "prefill"}]

    @pytest.mark.asyncio
    async def test_compaction_rewrite_drops_text_free_system_message(self):
        handler = AnthropicMessagesHandler()
        guardrail = MockCompactingGuardrail(
            replacement_messages=[
                {"role": "user", "content": "compacted history"},
                {"role": "system", "content": [{"type": "text", "text": ""}]},
                {"role": "system", "content": ""},
                {"role": "user", "content": "continue"},
            ]
        )
        guardrail.skip_system_message_in_guardrail = True
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": "original history"},
                {"role": "system", "content": "use the corrected result"},
                {"role": "user", "content": "continue"},
            ],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert [m["role"] for m in data["messages"]] == ["user", "user"]
        assert data["messages"][0]["content"] == [{"type": "text", "text": "compacted history"}]
        assert data["messages"][1]["content"] == [{"type": "text", "text": "continue"}]

    @pytest.mark.asyncio
    async def test_noncanonical_system_role_casing_is_still_scanned(self):
        handler = AnthropicMessagesHandler()
        guardrail = MockMaskingGuardrail()
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": "safe text"},
                {"role": "System", "content": "prohibited correction"},
            ],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.inputs is not None
        assert "prohibited correction" in guardrail.inputs["texts"]
        assert data["messages"][1]["content"] == "[MASKED]"

    @pytest.mark.asyncio
    async def test_midturn_system_keeps_tool_result_turns_aligned_for_masking(self):
        """Tool-result texts are scanned (LIT-5251), so counts align and the latest-user
        masking slice is locatable; a mid-turn system entry only shifts it by its own text."""
        from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import (
            BedrockGuardrail,
        )

        handler = AnthropicMessagesHandler()
        bedrock = BedrockGuardrail(guardrailIdentifier="gi", guardrailVersion="1")
        tool_loop = [
            {"role": "user", "content": "call the tool"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "tu_1", "name": "get", "input": {"a": 1}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu_1",
                        "content": [{"type": "text", "text": "tool output"}],
                    }
                ],
            },
        ]

        async def _slice_for(messages: list):
            guardrail = MockMaskingGuardrail()
            data = {"model": "claude-3-5-sonnet-20241022", "messages": messages}
            await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)
            assert guardrail.inputs is not None
            texts = guardrail.inputs["texts"]
            structured = guardrail.inputs["structured_messages"]
            target_index = bedrock._find_latest_message_index(structured, target_role="user")
            return (
                sum(bedrock._count_message_texts(m) for m in structured) - len(texts),
                bedrock._locate_message_texts_slice(
                    structured_messages=structured,
                    target_index=target_index,
                    texts=texts,
                ),
            )

        with_system = await _slice_for(
            tool_loop
            + [
                {"role": "system", "content": "use the corrected result"},
                {"role": "user", "content": "latest question"},
            ]
        )
        without_system = await _slice_for(tool_loop + [{"role": "user", "content": "latest question"}])

        assert with_system == (0, (3, 1))
        assert without_system == (0, (2, 1))

    @pytest.mark.asyncio
    async def test_compaction_rewrite_to_only_system_messages_is_rejected(self):
        import litellm

        handler = AnthropicMessagesHandler()
        guardrail = MockCompactingGuardrail(
            replacement_messages=[{"role": "system", "content": "use the corrected result"}]
        )
        guardrail.skip_system_message_in_guardrail = True
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": "original history"},
                {"role": "system", "content": "use the corrected result"},
            ],
        }

        with patch.object(litellm, "modify_params", False):
            with pytest.raises(litellm.BadRequestError, match="at least one non-system message"):
                await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

    @pytest.mark.asyncio
    async def test_compaction_rewrite_to_only_system_messages_repaired_with_modify_params(
        self,
    ):
        import litellm

        handler = AnthropicMessagesHandler()
        guardrail = MockCompactingGuardrail(
            replacement_messages=[{"role": "system", "content": "use the corrected result"}]
        )
        guardrail.skip_system_message_in_guardrail = True
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": "original history"},
                {"role": "system", "content": "use the corrected result"},
            ],
        }

        with patch.object(litellm, "modify_params", True):
            await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert data["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Please continue."}]}]
        assert data["system"] == [{"type": "text", "text": "use the corrected result"}]

    @pytest.mark.asyncio
    async def test_compaction_rewrite_without_system_messages_is_unchanged(self):
        handler = AnthropicMessagesHandler()
        guardrail = MockCompactingGuardrail(replacement_messages=[{"role": "user", "content": "compacted history"}])
        data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "b"},
                {"role": "user", "content": "c"},
            ],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert data["messages"] == [{"role": "user", "content": [{"type": "text", "text": "compacted history"}]}]

    @pytest.mark.asyncio
    async def test_process_output_streaming_response_empty_choices(self):
        """Test that streaming response with empty choices doesn't raise IndexError

        This test verifies the fix for the bug where accessing model_response.choices[0]
        would raise IndexError when the response has an empty choices list.
        """
        from litellm.types.utils import ModelResponse

        handler = AnthropicMessagesHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        # Create a mock response with empty choices
        mock_response = ModelResponse(
            id="msg_123",
            created=1234567890,
            model="claude-3",
            object="chat.completion",
            choices=[],  # Empty choices
        )

        # Mock _check_streaming_has_ended to return True (stream ended)
        # and _build_complete_streaming_response to return the mock response
        with (
            patch.object(handler, "_check_streaming_has_ended", return_value=True),
            patch(
                "litellm.llms.anthropic.chat.guardrail_translation.handler.AnthropicPassthroughLoggingHandler._build_complete_streaming_response",
                return_value=mock_response,
            ),
        ):
            responses_so_far = [b"data: some chunk"]

            # This should not raise IndexError
            result = await handler.process_output_streaming_response(
                responses_so_far=responses_so_far,
                guardrail_to_apply=guardrail,
                litellm_logging_obj=MagicMock(),
            )

            # Should return the responses unchanged
            assert result == responses_so_far

    @pytest.mark.asyncio
    async def test_process_output_streaming_response_with_valid_choices(self):
        """Test that streaming response with valid choices still works correctly"""
        from litellm.types.utils import Choices, Message, ModelResponse

        handler = AnthropicMessagesHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        # Create a mock response with valid choices
        mock_response = ModelResponse(
            id="msg_123",
            created=1234567890,
            model="claude-3",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content="Hello world",
                        role="assistant",
                    ),
                )
            ],
        )

        # Mock _check_streaming_has_ended to return True (stream ended)
        # and _build_complete_streaming_response to return the mock response
        with (
            patch.object(handler, "_check_streaming_has_ended", return_value=True),
            patch(
                "litellm.llms.anthropic.chat.guardrail_translation.handler.AnthropicPassthroughLoggingHandler._build_complete_streaming_response",
                return_value=mock_response,
            ),
        ):
            responses_so_far = [b"data: some chunk"]

            # This should process successfully
            result = await handler.process_output_streaming_response(
                responses_so_far=responses_so_far,
                guardrail_to_apply=guardrail,
                litellm_logging_obj=MagicMock(),
            )

            # Should return the responses
            assert result == responses_so_far

    @pytest.mark.asyncio
    async def test_process_output_streaming_response_stream_not_ended(self):
        """Test that streaming response falls back to text processing when stream hasn't ended"""
        handler = AnthropicMessagesHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        # Mock _check_streaming_has_ended to return False (stream not ended)
        with (
            patch.object(handler, "_check_streaming_has_ended", return_value=False),
            patch.object(
                handler, "get_streaming_string_so_far", return_value="partial text"
            ),
        ):
            responses_so_far = [b"data: some chunk"]

            # This should process successfully using text-based guardrail
            result = await handler.process_output_streaming_response(
                responses_so_far=responses_so_far,
                guardrail_to_apply=guardrail,
                litellm_logging_obj=MagicMock(),
            )

            # Should return the responses
            assert result == responses_so_far

    @pytest.mark.asyncio
    async def test_process_input_messages_with_anthropic_native_tools(self):
        """Test that Anthropic native tools (tool_search_tool_regex) are preserved correctly

        This test verifies the fix for the bug where Anthropic native tools like
        tool_search_tool_regex_20251119 were being converted to OpenAI format and then
        not properly converted back, causing API errors.

        The guardrail converts tools to OpenAI format for processing, then they need to be
        converted back to Anthropic format. Native Anthropic tools should be preserved as-is,
        while regular tools should be converted to type="custom".
        """
        handler = AnthropicMessagesHandler()
        guardrail = MockPassThroughGuardrail(guardrail_name="test")

        data = {
            "model": "claude-opus-4-6",
            "messages": [
                {"role": "user", "content": "What is the weather in San Francisco?"}
            ],
            "tools": [
                {
                    "type": "tool_search_tool_regex_20251119",
                    "name": "tool_search_tool_regex",
                },
                {
                    "name": "get_weather",
                    "description": "Get the weather at a specific location",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                    "defer_loading": True,
                },
            ],
        }

        result = await handler.process_input_messages(
            data=data, guardrail_to_apply=guardrail, litellm_logging_obj=MagicMock()
        )

        # Verify tools are in correct Anthropic format
        tools = result["tools"]
        assert len(tools) == 2

        # First tool should be preserved as Anthropic native tool
        assert tools[0]["type"] == "tool_search_tool_regex_20251119"
        assert tools[0]["name"] == "tool_search_tool_regex"

        # Second tool should be converted to Anthropic custom tool format
        assert tools[1]["type"] == "custom"
        assert tools[1]["name"] == "get_weather"
        assert tools[1]["description"] == "Get the weather at a specific location"
        assert "input_schema" in tools[1]


class ToolAppendingGuardrail(CustomGuardrail):
    """Guardrail that appends a new OpenAI-format function tool, mimicking a
    guardrail that injects a retrieval/recovery tool the model can later call."""

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        tools = list(inputs.get("tools") or [])
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "injected_tool",
                    "description": "injected by guardrail",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        )
        inputs["tools"] = tools
        return inputs


class TestAnthropicMessagesHandlerToolInjection:
    """A tool a guardrail injects in OpenAI format must survive the write-back
    to Anthropic format alongside the request's original tools."""

    @pytest.mark.asyncio
    async def test_injected_tool_survives_when_request_already_has_tools(self):
        handler = AnthropicMessagesHandler()
        guardrail = ToolAppendingGuardrail(guardrail_name="test")

        data = {
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get the weather at a specific location",
                    "input_schema": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                }
            ],
        }

        result = await handler.process_input_messages(
            data=data, guardrail_to_apply=guardrail, litellm_logging_obj=MagicMock()
        )

        names = [t.get("name") for t in result["tools"]]
        assert "get_weather" in names
        assert "injected_tool" in names

    @pytest.mark.asyncio
    async def test_injected_tool_survives_when_request_has_no_tools(self):
        handler = AnthropicMessagesHandler()
        guardrail = ToolAppendingGuardrail(guardrail_name="test")

        data = {
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "hi"}],
        }

        result = await handler.process_input_messages(
            data=data, guardrail_to_apply=guardrail, litellm_logging_obj=MagicMock()
        )

        assert [t.get("name") for t in result["tools"]] == ["injected_tool"]


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])


class TestAnthropicMessagesIncrementalScan:
    """PR #33278: only_scan_new_messages through the real /v1/messages translation
    handler (the path Claude Code uses). Encodes the wire payloads observed in the
    live validation against a real Bedrock guardrail.
    """

    def _bedrock_guardrail(self):
        from litellm.proxy.guardrails.guardrail_hooks.bedrock_guardrails import BedrockGuardrail

        return BedrockGuardrail(
            guardrail_name="bedrock-incremental-anthropic",
            guardrailIdentifier="test-guardrail",
            guardrailVersion="DRAFT",
            default_on=True,
            only_scan_new_messages=True,
        )

    def _data(self, messages, session_id):
        return {
            "model": "claude-sonnet-4-5",
            "messages": messages,
            "system": "You are a helpful geography assistant.",
            "litellm_session_id": session_id,
        }

    @pytest.mark.asyncio
    async def test_first_turn_scans_all_eligible_then_second_turn_scans_only_diff(self):
        from unittest.mock import AsyncMock, patch

        handler = AnthropicMessagesHandler()
        guardrail = self._bedrock_guardrail()
        sid = "anth-sess-diff"
        turn1 = [{"role": "user", "content": "What is the capital of France?"}]
        turn2 = turn1 + [
            {"role": "assistant", "content": "Paris."},
            {"role": "user", "content": "What is the capital of Germany?"},
        ]
        with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"action": "NONE", "output": [], "outputs": []}
            await handler.process_input_messages(
                data=self._data(turn1, sid), guardrail_to_apply=guardrail
            )
            assert mock_api.call_count == 1
            assert [m["content"] for m in mock_api.call_args.kwargs["messages"]] == [
                "What is the capital of France?"
            ]
            mock_api.reset_mock()
            await handler.process_input_messages(
                data=self._data(turn2, sid), guardrail_to_apply=guardrail
            )
            assert mock_api.call_count == 1
            assert [m["content"] for m in mock_api.call_args.kwargs["messages"]] == [
                "Paris.",
                "What is the capital of Germany?",
            ]

    @pytest.mark.asyncio
    async def test_identical_resend_makes_no_guardrail_call(self):
        from unittest.mock import AsyncMock, patch

        handler = AnthropicMessagesHandler()
        guardrail = self._bedrock_guardrail()
        sid = "anth-sess-resend"
        msgs = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris."},
            {"role": "user", "content": "What is the capital of Germany?"},
        ]
        with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"action": "NONE", "output": [], "outputs": []}
            await handler.process_input_messages(data=self._data(msgs, sid), guardrail_to_apply=guardrail)
            assert mock_api.call_count == 1
            mock_api.reset_mock()
            await handler.process_input_messages(data=self._data(msgs, sid), guardrail_to_apply=guardrail)
            mock_api.assert_not_called()

    @pytest.mark.asyncio
    async def test_edited_history_message_is_rescanned(self):
        from unittest.mock import AsyncMock, patch

        handler = AnthropicMessagesHandler()
        guardrail = self._bedrock_guardrail()
        sid = "anth-sess-edit"
        msgs = [{"role": "user", "content": "What is the capital of France?"}]
        edited = [{"role": "user", "content": "What is the capital and population of France?"}]
        with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"action": "NONE", "output": [], "outputs": []}
            await handler.process_input_messages(data=self._data(msgs, sid), guardrail_to_apply=guardrail)
            mock_api.reset_mock()
            await handler.process_input_messages(data=self._data(edited, sid), guardrail_to_apply=guardrail)
            assert mock_api.call_count == 1
            assert [m["content"] for m in mock_api.call_args.kwargs["messages"]] == [
                "What is the capital and population of France?"
            ]

    @pytest.mark.asyncio
    async def test_mixed_text_and_tool_use_keeps_text_segments(self):
        """A message carrying both text and a tool_use block must not lose its text.
        (tool_use inputs are still dropped from texts on the anthropic input path;
        tool_result content is scanned, see TestAnthropicMessagesToolResultScanning.)"""
        from unittest.mock import AsyncMock, patch

        handler = AnthropicMessagesHandler()
        guardrail = self._bedrock_guardrail()
        sid = "anth-sess-tools"
        msgs = [
            {"role": "user", "content": "Search for the weather in Paris"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me look that up for you."},
                    {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"query": "canary-args"}},
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "canary-result"}],
            },
            {"role": "user", "content": "Thanks, summarize the result."},
        ]
        with patch.object(guardrail, "make_bedrock_api_request", new_callable=AsyncMock) as mock_api:
            mock_api.return_value = {"action": "NONE", "output": [], "outputs": []}
            await handler.process_input_messages(data=self._data(msgs, sid), guardrail_to_apply=guardrail)
            scanned = [m["content"] for m in mock_api.call_args.kwargs["messages"]]
            assert "Let me look that up for you." in scanned, "text beside a tool_use must be scanned"
            assert "Search for the weather in Paris" in scanned
            assert "Thanks, summarize the result." in scanned


class MockCanaryMaskingGuardrail(CustomGuardrail):
    """Records every text handed to it and masks a canary token in place."""

    def __init__(self, guardrail_name: str = "mask-canary"):
        super().__init__(guardrail_name=guardrail_name)
        self.seen_texts: list[str] = []

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        texts = list(inputs.get("texts") or [])
        self.seen_texts.extend(texts)
        inputs["texts"] = [t.replace("POISON", "[BLOCKED]") for t in texts]
        return inputs


class TestAnthropicMessagesToolResultScanning:
    """LIT-5251: tool_result blocks carry whatever a client's local tool fetched, so
    they are the request-path payload an indirect prompt injection actually arrives in.
    Both wire shapes Anthropic accepts must be scanned and rewritten in place.
    """

    def _data(self, messages):
        return {"model": "claude-sonnet-4-5", "messages": messages}

    @pytest.mark.asyncio
    async def test_string_form_tool_result_is_scanned_and_written_back(self):
        handler = AnthropicMessagesHandler()
        guardrail = MockCanaryMaskingGuardrail()
        messages = [
            {"role": "user", "content": "fetch the page"},
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "tu1", "name": "Bash", "input": {"cmd": "curl"}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "page says POISON here"}],
            },
        ]

        await handler.process_input_messages(data=self._data(messages), guardrail_to_apply=guardrail)

        assert "page says POISON here" in guardrail.seen_texts, "string-form tool_result must reach the guardrail"
        assert messages[2]["content"][0]["content"] == "page says [BLOCKED] here", (
            "masked text must be written back into the tool_result, not dropped"
        )

    @pytest.mark.asyncio
    async def test_list_form_tool_result_is_scanned_and_written_back(self):
        handler = AnthropicMessagesHandler()
        guardrail = MockCanaryMaskingGuardrail()
        messages = [
            {"role": "user", "content": "fetch the page"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu1",
                        "content": [
                            {"type": "text", "text": "first POISON block"},
                            {"type": "text", "text": "second POISON block"},
                        ],
                    }
                ],
            },
        ]

        await handler.process_input_messages(data=self._data(messages), guardrail_to_apply=guardrail)

        assert "first POISON block" in guardrail.seen_texts
        assert "second POISON block" in guardrail.seen_texts
        blocks = messages[1]["content"][0]["content"]
        assert blocks[0]["text"] == "first [BLOCKED] block"
        assert blocks[1]["text"] == "second [BLOCKED] block"

    @pytest.mark.asyncio
    async def test_write_back_targets_stay_aligned_across_mixed_shapes(self):
        """The write-back is positional, so a single mis-indexed target silently
        writes one message's masked text over another's."""
        handler = AnthropicMessagesHandler()
        guardrail = MockCanaryMaskingGuardrail()
        messages = [
            {"role": "user", "content": "plain POISON string"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "sibling POISON text"},
                    {"type": "tool_result", "tool_use_id": "tu1", "content": "string POISON result"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu2",
                        "content": [{"type": "text", "text": "nested POISON result"}],
                    },
                ],
            },
            {"role": "user", "content": "trailing POISON string"},
        ]

        await handler.process_input_messages(data=self._data(messages), guardrail_to_apply=guardrail)

        assert messages[0]["content"] == "plain [BLOCKED] string"
        assert messages[1]["content"][0]["text"] == "sibling [BLOCKED] text"
        assert messages[1]["content"][1]["content"] == "string [BLOCKED] result"
        assert messages[1]["content"][2]["content"][0]["text"] == "nested [BLOCKED] result"
        assert messages[2]["content"] == "trailing [BLOCKED] string"

    @pytest.mark.asyncio
    async def test_image_inside_tool_result_is_collected(self):
        handler = AnthropicMessagesHandler()

        class ImageRecordingGuardrail(MockCanaryMaskingGuardrail):
            def __init__(self):
                super().__init__()
                self.seen_images: list[str] = []

            async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
                self.seen_images.extend(inputs.get("images") or [])
                return await super().apply_guardrail(inputs, request_data, input_type, logging_obj)

        guardrail = ImageRecordingGuardrail()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tu1",
                        "content": [
                            {"type": "text", "text": "screenshot POISON"},
                            {"type": "image", "source": {"type": "base64", "data": "SCREENSHOT_BYTES"}},
                        ],
                    }
                ],
            }
        ]

        await handler.process_input_messages(data=self._data(messages), guardrail_to_apply=guardrail)

        assert "SCREENSHOT_BYTES" in guardrail.seen_images, "images nested in a tool_result must be scanned too"

    @pytest.mark.asyncio
    async def test_tool_result_is_skipped_when_guardrail_skips_tool_messages(self):
        handler = AnthropicMessagesHandler()
        guardrail = MockCanaryMaskingGuardrail()
        guardrail.skip_tool_message_in_guardrail = True
        messages = [
            {"role": "user", "content": "keep me POISON"},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "skip me POISON"}],
            },
        ]

        await handler.process_input_messages(data=self._data(messages), guardrail_to_apply=guardrail)

        assert "skip me POISON" not in guardrail.seen_texts
        assert messages[1]["content"][0]["content"] == "skip me POISON"
        assert messages[0]["content"] == "keep me [BLOCKED]"


class InputsRecordingGuardrail(MockCanaryMaskingGuardrail):
    def __init__(self):
        super().__init__(guardrail_name="scan-only-capture")
        self.captured_inputs: Optional[GenericGuardrailAPIInputs] = None

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        self.captured_inputs = inputs
        return await super().apply_guardrail(inputs, request_data, input_type, logging_obj)


class StructuredMessagesRewritingGuardrail(CustomGuardrail):
    """Returns a new structured_messages list with a canary redacted, like redaction guardrails do."""

    def __init__(self):
        super().__init__(guardrail_name="structured-rewrite")

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional[Any] = None,
    ) -> GenericGuardrailAPIInputs:
        structured = inputs.get("structured_messages") or []
        inputs["structured_messages"] = [
            json.loads(json.dumps(message).replace("POISON", "[BLOCKED]")) for message in structured
        ]
        return inputs


class TestAnthropicMessagesScanOnlyToolResults:
    def _guardrail(self):
        guardrail = InputsRecordingGuardrail()
        guardrail.scan_only_tool_results = True
        return guardrail

    @pytest.mark.asyncio
    async def test_structured_write_back_merges_into_the_full_conversation(self):
        handler = AnthropicMessagesHandler()
        guardrail = StructuredMessagesRewritingGuardrail()
        guardrail.scan_only_tool_results = True
        data = {
            "model": "claude-sonnet-4-5",
            "system": "You are a careful agent harness.",
            "messages": [
                {"role": "user", "content": "fetch the page"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "tu1", "name": "Bash", "input": {"cmd": "curl"}}],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "fetched POISON page"}],
                },
            ],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert data["system"] == "You are a careful agent harness."
        assert [m["role"] for m in data["messages"]] == ["user", "assistant", "user"], (
            "a redacting guardrail must not strip out-of-scope turns from the request"
        )
        serialized = json.dumps(data["messages"])
        assert "fetch the page" in serialized
        assert "tool_use" in serialized
        assert "fetched [BLOCKED] page" in serialized
        assert "POISON" not in serialized

    @pytest.mark.asyncio
    async def test_scan_narrows_to_tool_results_and_write_back_stays_aligned(self):
        handler = AnthropicMessagesHandler()
        guardrail = self._guardrail()
        data = {
            "model": "claude-sonnet-4-5",
            "system": "You are a trusted agent harness with POISON heuristics.",
            "tools": [
                {
                    "name": "Bash",
                    "description": "run a command",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            "messages": [
                {"role": "user", "content": "scaffolding POISON prompt"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "tu1", "name": "Bash", "input": {"cmd": "curl"}}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "sibling POISON text"},
                        {"type": "tool_result", "tool_use_id": "tu1", "content": "fetched POISON page"},
                    ],
                },
            ],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.seen_texts == ["fetched POISON page"], (
            "only the tool_result payload may reach the guardrail"
        )
        assert guardrail.captured_inputs is not None
        assert guardrail.captured_inputs.get("tools") is None
        assert [m["role"] for m in guardrail.captured_inputs["structured_messages"]] == ["tool"]
        assert data["messages"][2]["content"][1]["content"] == "fetched [BLOCKED] page"
        assert data["messages"][0]["content"] == "scaffolding POISON prompt", (
            "out-of-scope content must come back untouched, not masked or dropped"
        )
        assert data["messages"][2]["content"][0]["text"] == "sibling POISON text"

    @pytest.mark.asyncio
    async def test_guardrail_synthesized_tools_are_appended_without_replacing_request_tools(self):
        handler = AnthropicMessagesHandler()
        guardrail = ToolAppendingGuardrail(guardrail_name="tool-appending")
        guardrail.scan_only_tool_results = True
        original_tools = [
            {
                "name": "get_weather",
                "description": "Get the weather at a specific location",
                "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}},
            }
        ]
        data = {
            "model": "claude-sonnet-4-5",
            "tools": original_tools,
            "messages": [
                {"role": "user", "content": "what's the weather?"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "tu1", "name": "get_weather", "input": {}}],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "sunny"}],
                },
            ],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert [t["name"] for t in data["tools"]] == ["get_weather", "injected_tool"], (
            "a tool the guardrail synthesized must reach the model, converted to Anthropic format, "
            "without the request's own tools being replaced or dropped"
        )
        assert data["tools"][0] == original_tools[0]

    @pytest.mark.asyncio
    async def test_guardrail_is_not_called_when_the_request_has_no_tool_results(self):
        handler = AnthropicMessagesHandler()
        guardrail = self._guardrail()
        data = {
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "What is 2 plus 2?"}],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.captured_inputs is None
        assert guardrail.seen_texts == []

    @pytest.mark.asyncio
    async def test_images_are_scoped_the_same_way_as_texts(self):
        handler = AnthropicMessagesHandler()
        guardrail = self._guardrail()
        data = {
            "model": "claude-sonnet-4-5",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "image", "source": {"type": "base64", "data": "USER_IMG"}}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu1",
                            "content": [
                                {"type": "text", "text": "screenshot POISON"},
                                {"type": "image", "source": {"type": "base64", "data": "TOOL_IMG"}},
                            ],
                        }
                    ],
                },
            ],
        }

        await handler.process_input_messages(data=data, guardrail_to_apply=guardrail)

        assert guardrail.captured_inputs is not None
        assert guardrail.captured_inputs.get("images") == ["TOOL_IMG"]
