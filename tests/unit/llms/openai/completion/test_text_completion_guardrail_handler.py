"""
Unit tests for OpenAI Text Completion Guardrail Translation Handler
"""

import json
from typing import List, Literal, Optional, Tuple
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException


from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms import get_guardrail_translation_mapping
from litellm.llms.openai.completion.guardrail_translation.handler import (
    OpenAITextCompletionHandler,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import ContentFilterGuardrail
from litellm.proxy.guardrails.guardrail_hooks.unified_guardrail.unified_guardrail import UnifiedLLMGuardrails
from litellm.types.guardrails import BlockedWord, ContentFilterAction, GuardrailEventHooks
from litellm.types.utils import CallTypes, GenericGuardrailAPIInputs, TextChoices, TextCompletionResponse, Usage


class MockGuardrail(CustomGuardrail):
    """Mock guardrail for testing"""

    async def apply_guardrail(
        self, inputs: dict, request_data: dict, input_type: str, **kwargs
    ) -> dict:
        texts = inputs.get("texts", [])
        return {"texts": [f"{text} [GUARDRAILED]" for text in texts]}


class TestHandlerDiscovery:
    """Test that the handler is properly discovered"""

    def test_handler_discovered_for_text_completion(self):
        """Test that text_completion CallType is mapped to handler"""
        handler_class = get_guardrail_translation_mapping(CallTypes.text_completion)
        assert handler_class == OpenAITextCompletionHandler

    def test_handler_discovered_for_atext_completion(self):
        """Test that atext_completion CallType is mapped to handler"""
        handler_class = get_guardrail_translation_mapping(CallTypes.atext_completion)
        assert handler_class == OpenAITextCompletionHandler


class TestInputProcessing:
    """Test input processing functionality"""

    @pytest.mark.asyncio
    async def test_process_simple_string_prompt(self):
        """Test processing a simple string prompt"""
        handler = OpenAITextCompletionHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {
            "model": "gpt-3.5-turbo-instruct",
            "prompt": "Say this is a test",
            "max_tokens": 7,
        }

        result = await handler.process_input_messages(data, guardrail)

        assert result["prompt"] == "Say this is a test [GUARDRAILED]"
        assert result["model"] == "gpt-3.5-turbo-instruct"
        assert result["max_tokens"] == 7

    @pytest.mark.asyncio
    async def test_process_list_of_string_prompts(self):
        """Test processing a list of string prompts (batch completion)"""
        handler = OpenAITextCompletionHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {
            "model": "gpt-3.5-turbo-instruct",
            "prompt": ["Tell me a joke", "Write a poem", "Say hello"],
            "max_tokens": 50,
        }

        result = await handler.process_input_messages(data, guardrail)

        assert result["prompt"] == [
            "Tell me a joke [GUARDRAILED]",
            "Write a poem [GUARDRAILED]",
            "Say hello [GUARDRAILED]",
        ]
        assert result["model"] == "gpt-3.5-turbo-instruct"

    @pytest.mark.asyncio
    async def test_process_no_prompt(self):
        """Test processing when no prompt is provided"""
        handler = OpenAITextCompletionHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {"model": "gpt-3.5-turbo-instruct", "max_tokens": 7}

        result = await handler.process_input_messages(data, guardrail)

        # Should return data unchanged when no prompt
        assert result == data
        assert "prompt" not in result

    @pytest.mark.asyncio
    async def test_process_empty_prompt(self):
        """Test processing when prompt is None"""
        handler = OpenAITextCompletionHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {"model": "gpt-3.5-turbo-instruct", "prompt": None}

        result = await handler.process_input_messages(data, guardrail)

        # Should return data unchanged when prompt is None
        assert result["prompt"] is None

    @pytest.mark.asyncio
    async def test_process_mixed_list_prompts(self):
        """Test processing a list with non-string items (e.g., token lists)"""
        handler = OpenAITextCompletionHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {
            "model": "gpt-3.5-turbo-instruct",
            "prompt": ["String prompt", [1, 2, 3], "Another string"],
            "max_tokens": 50,
        }

        result = await handler.process_input_messages(data, guardrail)

        # String items should be guardrailed, list items unchanged
        assert result["prompt"] == [
            "String prompt [GUARDRAILED]",
            [1, 2, 3],  # Unchanged
            "Another string [GUARDRAILED]",
        ]

    @pytest.mark.asyncio
    async def test_process_long_prompt(self):
        """Test processing a long prompt"""
        handler = OpenAITextCompletionHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        long_prompt = "This is a very long prompt. " * 50

        data = {"model": "gpt-3.5-turbo-instruct", "prompt": long_prompt}

        result = await handler.process_input_messages(data, guardrail)

        assert result["prompt"] == f"{long_prompt} [GUARDRAILED]"
        assert long_prompt in result["prompt"]


class TestOutputProcessing:
    """Test output processing functionality"""

    @pytest.mark.asyncio
    async def test_process_output_single_choice(self):
        """Test processing output with a single choice"""
        handler = OpenAITextCompletionHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        # Create a mock TextCompletionResponse
        choice = TextChoices(
            finish_reason="stop",
            index=0,
            logprobs=None,
            text="This is indeed a test",
        )
        response = TextCompletionResponse(
            id="cmpl-test",
            choices=[choice],
            created=1589478378,
            model="gpt-3.5-turbo-instruct",
            object="text_completion",
        )

        result = await handler.process_output_response(response, guardrail)

        # Verify guardrail was applied to output
        assert result.choices[0].text == "This is indeed a test [GUARDRAILED]"
        assert result.id == "cmpl-test"
        assert result.model == "gpt-3.5-turbo-instruct"

    @pytest.mark.asyncio
    async def test_process_output_multiple_choices(self):
        """Test processing output with multiple choices (n > 1)"""
        handler = OpenAITextCompletionHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        # Create mock choices
        choices = [
            TextChoices(
                finish_reason="stop",
                index=0,
                logprobs=None,
                text="Response 1",
            ),
            TextChoices(
                finish_reason="stop",
                index=1,
                logprobs=None,
                text="Response 2",
            ),
        ]
        response = TextCompletionResponse(
            id="cmpl-test",
            choices=choices,
            created=1589478378,
            model="gpt-3.5-turbo-instruct",
            object="text_completion",
        )

        result = await handler.process_output_response(response, guardrail)

        # Verify guardrail was applied to all choices
        assert result.choices[0].text == "Response 1 [GUARDRAILED]"
        assert result.choices[1].text == "Response 2 [GUARDRAILED]"

    @pytest.mark.asyncio
    async def test_process_output_empty_choices(self):
        """Test processing output with no choices"""
        handler = OpenAITextCompletionHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        response = TextCompletionResponse(
            id="cmpl-test",
            choices=[],
            created=1589478378,
            model="gpt-3.5-turbo-instruct",
            object="text_completion",
        )

        result = await handler.process_output_response(response, guardrail)

        # Should return unchanged when no choices
        assert result == response
        assert len(result.choices) == 0


class TestPIIMaskingScenario:
    """Test real-world scenario: PII masking in prompts and completions"""

    @pytest.mark.asyncio
    async def test_pii_masking_in_prompt_and_completion(self):
        """Test that PII can be masked from both input prompt and output completion"""

        class PIIMaskingGuardrail(CustomGuardrail):
            """Mock PII masking guardrail"""

            async def apply_guardrail(
                self, inputs: dict, request_data: dict, input_type: str, **kwargs
            ) -> dict:
                # Simple mock: replace email-like patterns
                import re

                texts = inputs.get("texts", [])
                masked_texts = []
                for text in texts:
                    masked = re.sub(
                        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                        "[EMAIL_REDACTED]",
                        text,
                    )
                    # Replace names (simple mock)
                    masked = masked.replace("John Doe", "[NAME_REDACTED]")
                    masked_texts.append(masked)
                return {"texts": masked_texts}

        handler = OpenAITextCompletionHandler()
        guardrail = PIIMaskingGuardrail(guardrail_name="mask_pii")

        # Test input processing
        data = {
            "model": "gpt-3.5-turbo-instruct",
            "prompt": "My name is John Doe and my email is john@example.com",
        }

        result = await handler.process_input_messages(data, guardrail)

        # Verify PII was masked in input
        assert "john@example.com" not in result["prompt"]
        assert "John Doe" not in result["prompt"]
        assert "[EMAIL_REDACTED]" in result["prompt"]
        assert "[NAME_REDACTED]" in result["prompt"]

        # Test output processing
        choice = TextChoices(
            finish_reason="stop",
            index=0,
            logprobs=None,
            text="You can reach me at admin@company.com",
        )
        response = TextCompletionResponse(
            id="cmpl-test",
            choices=[choice],
            created=1589478378,
            model="gpt-3.5-turbo-instruct",
            object="text_completion",
        )

        output_result = await handler.process_output_response(response, guardrail)

        # Verify PII was masked in output
        assert "admin@company.com" not in output_result.choices[0].text
        assert "[EMAIL_REDACTED]" in output_result.choices[0].text

    @pytest.mark.asyncio
    async def test_batch_prompts_with_pii(self):
        """Test PII masking with batch prompts"""

        class PIIMaskingGuardrail(CustomGuardrail):
            """Mock PII masking guardrail"""

            async def apply_guardrail(
                self, inputs: dict, request_data: dict, input_type: str, **kwargs
            ) -> dict:
                import re

                texts = inputs.get("texts", [])
                masked_texts = []
                for text in texts:
                    masked = re.sub(
                        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                        "[EMAIL_REDACTED]",
                        text,
                    )
                    masked_texts.append(masked)
                return {"texts": masked_texts}

        handler = OpenAITextCompletionHandler()
        guardrail = PIIMaskingGuardrail(guardrail_name="mask_pii")

        data = {
            "model": "gpt-3.5-turbo-instruct",
            "prompt": [
                "Contact me at alice@example.com",
                "Send to bob@test.org",
            ],
        }

        result = await handler.process_input_messages(data, guardrail)

        # Verify all prompts had PII masked
        assert "[EMAIL_REDACTED]" in result["prompt"][0]
        assert "[EMAIL_REDACTED]" in result["prompt"][1]
        assert "alice@example.com" not in result["prompt"][0]
        assert "bob@test.org" not in result["prompt"][1]


class _RecordingKeywordFilter(ContentFilterGuardrail):
    def __init__(self):
        super().__init__(
            guardrail_name="keyword-filter",
            blocked_words=[
                BlockedWord(keyword="persimmon", action=ContentFilterAction.MASK),
                BlockedWord(keyword="kumquat", action=ContentFilterAction.BLOCK),
            ],
            event_hook=GuardrailEventHooks.post_call,
            default_on=True,
        )
        self.streaming_sampling_rate = 1
        self.scanned_texts: tuple[tuple[str, ...], ...] = ()

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        self.scanned_texts = (*self.scanned_texts, tuple(inputs.get("texts") or ()))
        return await super().apply_guardrail(inputs, request_data, input_type, logging_obj)


class _PassthroughReplaceGuardrail(CustomGuardrail):
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        if any("kumquat" in text for text in inputs.get("texts") or []):
            self.raise_passthrough_exception(violation_message="replaced kumquat", request_data=request_data)
        return inputs


def _text_chunk(index: int | None, text: str | None, finish_reason: str | None = None) -> TextCompletionResponse:
    return TextCompletionResponse(
        id="cmpl-stream",
        created=1589478378,
        model="gpt-3.5-turbo-instruct",
        object="text_completion",
        choices=[TextChoices(index=index, text=text, finish_reason=finish_reason)],
    )


def _usage_chunk() -> TextCompletionResponse:
    return TextCompletionResponse(
        id="cmpl-stream",
        created=1589478378,
        model="gpt-3.5-turbo-instruct",
        object="text_completion",
        choices=[],
        usage=Usage(prompt_tokens=3, completion_tokens=5, total_tokens=8),
    )


def _delivered(chunks: list[TextCompletionResponse]) -> list[tuple[int, str | None, str | None]]:
    return [(choice.index, choice.text, choice.finish_reason) for chunk in chunks for choice in chunk.choices]


async def _drive_flat_stream(guardrail: CustomGuardrail, chunks: list[TextCompletionResponse]) -> list[object]:
    async def upstream():
        for chunk in chunks:
            yield chunk

    return [
        item
        async for item in UnifiedLLMGuardrails().async_post_call_streaming_iterator_hook(
            user_api_key_dict=UserAPIKeyAuth(api_key="test-key", request_route="/v1/completions"),
            response=upstream(),
            request_data={"guardrail_to_apply": guardrail, "model": "gpt-3.5-turbo-instruct"},
        )
    ]


class TestStreamingOutputProcessing:
    @pytest.mark.asyncio
    async def test_ended_stream_delivers_each_choice_masked_text(self):
        guardrail = _RecordingKeywordFilter()
        chunks = [
            _text_chunk(0, "A pers"),
            _text_chunk(1, "kiwi "),
            _text_chunk(0, "im"),
            _text_chunk(1, "pie", "stop"),
            _text_chunk(0, "mon tart"),
            _text_chunk(0, None, "length"),
            _usage_chunk(),
        ]

        await OpenAITextCompletionHandler().process_output_streaming_response(
            responses_so_far=chunks,
            guardrail_to_apply=guardrail,
            request_data={},
            deliver_ended_stream_rewrites=True,
        )

        assert guardrail.scanned_texts == (("A persimmon tart", "kiwi pie"),)
        assert _delivered(chunks) == [
            (0, "A [KEYWORD_REDACTED] tart", None),
            (1, "kiwi ", None),
            (0, "", None),
            (1, "pie", "stop"),
            (0, "", None),
            (0, None, "length"),
        ]
        assert chunks[-1].usage.total_tokens == 8

    @pytest.mark.asyncio
    async def test_ended_stream_scanned_without_delivery_keeps_chunks_and_stores_the_assembled_response(self):
        chunks = [_text_chunk(0, "A pers"), _text_chunk(0, "immon tart"), _text_chunk(0, "", "stop"), _usage_chunk()]
        request_data: dict[str, object] = {}

        await OpenAITextCompletionHandler().process_output_streaming_response(
            responses_so_far=chunks, guardrail_to_apply=_RecordingKeywordFilter(), request_data=request_data
        )

        assert _delivered(chunks) == [(0, "A pers", None), (0, "immon tart", None), (0, "", "stop")]
        assembled = request_data["response"]
        assert isinstance(assembled, TextCompletionResponse)
        assert [(c.text, c.finish_reason) for c in assembled.choices] == [("A [KEYWORD_REDACTED] tart", "stop")]
        assert assembled.usage.total_tokens == 8

    @pytest.mark.asyncio
    async def test_ended_stream_block_raises_the_guardrail_error(self):
        chunks = [_text_chunk(0, "A kum"), _text_chunk(0, "quat tart"), _text_chunk(0, "", "stop")]

        with pytest.raises(HTTPException) as exc_info:
            await OpenAITextCompletionHandler().process_output_streaming_response(
                responses_so_far=chunks, guardrail_to_apply=_RecordingKeywordFilter(), deliver_ended_stream_rewrites=True
            )

        assert exc_info.value.status_code == 400
        assert "kumquat" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_ended_stream_treats_a_null_choice_index_as_choice_zero(self):
        guardrail = _RecordingKeywordFilter()
        chunks = [_text_chunk(None, "A pers"), _text_chunk(0, "immon tart"), _text_chunk(0, "", "stop")]

        await OpenAITextCompletionHandler().process_output_streaming_response(
            responses_so_far=chunks, guardrail_to_apply=guardrail, request_data={}, deliver_ended_stream_rewrites=True
        )

        assert guardrail.scanned_texts == (("A persimmon tart",),)
        assert [choice.text for chunk in chunks for choice in chunk.choices] == ["A [KEYWORD_REDACTED] tart", "", ""]

    @pytest.mark.asyncio
    async def test_ended_stream_blocks_a_keyword_split_across_a_null_and_a_zero_choice_index(self):
        chunks = [_text_chunk(None, "A kum"), _text_chunk(0, "quat tart"), _text_chunk(0, "", "stop")]

        with pytest.raises(HTTPException) as exc_info:
            await OpenAITextCompletionHandler().process_output_streaming_response(
                responses_so_far=chunks, guardrail_to_apply=_RecordingKeywordFilter(), deliver_ended_stream_rewrites=True
            )

        assert exc_info.value.status_code == 400
        assert "kumquat" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_flat_stream_is_scanned_once_after_it_ends(self):
        guardrail = _RecordingKeywordFilter()
        chunks = [_text_chunk(0, "A pers"), _text_chunk(0, "immon tart"), _text_chunk(0, "", "stop"), _usage_chunk()]

        delivered = await _drive_flat_stream(guardrail, chunks)

        assert guardrail.scanned_texts == (("A persimmon tart",),)
        assert len(delivered) == len(chunks)

    @pytest.mark.asyncio
    async def test_flat_stream_block_ends_with_an_in_band_error_frame(self):
        chunks = [_text_chunk(0, "A kum"), _text_chunk(0, "quat tart"), _text_chunk(0, "", "stop"), _usage_chunk()]

        delivered = await _drive_flat_stream(_RecordingKeywordFilter(), chunks)

        error_frame = delivered[-1]
        assert isinstance(error_frame, bytes)
        assert error_frame.startswith(b'data: {"error": ')
        assert b"kumquat" in error_frame

    @pytest.mark.asyncio
    async def test_flat_stream_passthrough_block_appends_the_replacement_and_ends_with_content_filter(self):
        guardrail = _PassthroughReplaceGuardrail(
            guardrail_name="passthrough", event_hook=GuardrailEventHooks.post_call, default_on=True
        )
        chunks = [_text_chunk(0, "A kum"), _text_chunk(0, "quat tart"), _text_chunk(0, "", "stop"), _usage_chunk()]

        delivered = await _drive_flat_stream(guardrail, chunks)

        frames = [json.loads(item.decode().removeprefix("data: ")) for item in delivered if isinstance(item, bytes)]
        assert [(frame["choices"][0]["text"], frame["choices"][0].get("finish_reason")) for frame in frames] == [
            ("replaced kumquat", None),
            ("", "content_filter"),
        ]
        assert frames[-1]["usage"]["total_tokens"] == 8

    @pytest.mark.asyncio
    async def test_flat_stream_passthrough_block_reports_no_usage_the_upstream_never_sent(self):
        guardrail = _PassthroughReplaceGuardrail(
            guardrail_name="passthrough", event_hook=GuardrailEventHooks.post_call, default_on=True
        )
        chunks = [_text_chunk(0, "A kum"), _text_chunk(0, "quat tart"), _text_chunk(0, "", "stop")]

        delivered = await _drive_flat_stream(guardrail, [chunk.model_copy(update={"usage": None}) for chunk in chunks])

        frames = [json.loads(item.decode().removeprefix("data: ")) for item in delivered if isinstance(item, bytes)]
        assert frames[-1]["choices"][0]["finish_reason"] == "content_filter"
        assert [frame for frame in frames if "usage" in frame] == []
