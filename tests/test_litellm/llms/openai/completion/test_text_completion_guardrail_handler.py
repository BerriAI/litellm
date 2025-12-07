"""
Unit tests for OpenAI Text Completion Guardrail Translation Handler
"""

import os
import sys
from typing import List, Optional, Tuple
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms import get_guardrail_translation_mapping
from litellm.llms.openai.completion.guardrail_translation.handler import (
    OpenAITextCompletionHandler,
)
from litellm.types.utils import CallTypes, TextChoices, TextCompletionResponse


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
