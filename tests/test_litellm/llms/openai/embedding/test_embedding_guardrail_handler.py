"""
Unit tests for OpenAI Embedding Guardrail Translation Handler

Tests the handler's ability to process embedding inputs with guardrail transformations.
Covers single strings, lists of strings, length limits, and reconstruction logic.
"""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms import get_guardrail_translation_mapping
from litellm.llms.openai.embedding.guardrail_translation.handler import (
    OpenAIEmbeddingHandler,
)
from litellm.types.utils import CallTypes
from litellm.utils import EmbeddingResponse


class MockGuardrail(CustomGuardrail):
    """Mock guardrail for testing that transforms text"""

    async def apply_guardrail(self, text: str) -> str:
        """Append [GUARDRAILED] to text"""
        return f"{text} [GUARDRAILED]"


class TestHandlerDiscovery:
    """Test that the handler is properly discovered by the guardrail system"""

    def test_handler_discovered_for_embedding(self):
        """Test that handler is discovered for CallTypes.embedding"""
        handler_class = get_guardrail_translation_mapping(CallTypes.embedding)
        assert handler_class == OpenAIEmbeddingHandler

    def test_handler_discovered_for_aembedding(self):
        """Test that handler is discovered for CallTypes.aembedding"""
        handler_class = get_guardrail_translation_mapping(CallTypes.aembedding)
        assert handler_class == OpenAIEmbeddingHandler

    def test_handler_has_required_methods(self):
        """Test that handler has required methods"""
        handler = OpenAIEmbeddingHandler()
        assert hasattr(handler, "process_input_messages")
        assert hasattr(handler, "process_output_response")
        assert callable(handler.process_input_messages)
        assert callable(handler.process_output_response)


class TestInputProcessingSingleString:
    """Test input processing for single string inputs"""

    @pytest.mark.asyncio
    async def test_process_single_string(self):
        """Test processing a single string input"""
        handler = OpenAIEmbeddingHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {
            "model": "text-embedding-ada-002",
            "input": "The quick brown fox jumps over the lazy dog",
        }

        result = await handler.process_input_messages(data, guardrail)

        assert (
            result["input"]
            == "The quick brown fox jumps over the lazy dog [GUARDRAILED]"
        )
        assert result["model"] == "text-embedding-ada-002"

    @pytest.mark.asyncio
    async def test_process_empty_string(self):
        """Test processing an empty string"""
        handler = OpenAIEmbeddingHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {"model": "text-embedding-ada-002", "input": ""}

        result = await handler.process_input_messages(data, guardrail)

        assert result["input"] == " [GUARDRAILED]"

    @pytest.mark.asyncio
    async def test_process_no_input(self):
        """Test processing when no input is provided"""
        handler = OpenAIEmbeddingHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {"model": "text-embedding-ada-002"}

        result = await handler.process_input_messages(data, guardrail)

        # Should return data unchanged when no input
        assert result == data
        assert "input" not in result

    @pytest.mark.asyncio
    async def test_process_long_string(self):
        """Test processing a long string"""
        handler = OpenAIEmbeddingHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        long_text = "This is a test sentence. " * 50

        data = {"model": "text-embedding-ada-002", "input": long_text}

        result = await handler.process_input_messages(data, guardrail)

        assert result["input"] == f"{long_text} [GUARDRAILED]"
        assert long_text in result["input"]

    @pytest.mark.asyncio
    @patch(
        "litellm.llms.openai.embedding.guardrail_translation.handler.MAX_GUARDRAIL_INPUT_LENGTH",
        10,
    )
    @patch(
        "litellm.llms.openai.embedding.guardrail_translation.handler.DISABLE_MAX_GUARDRAIL_INPUT_CHECK",
        False,
    )
    async def test_process_single_string_exceeds_length_limit(self):
        """Test that single string exceeding length limit skips guardrail"""
        handler = OpenAIEmbeddingHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        long_text = "This is a very long text that exceeds the limit"

        data = {"model": "text-embedding-ada-002", "input": long_text}

        result = await handler.process_input_messages(data, guardrail)

        # Should keep original text (not guardrailed) when exceeds limit
        assert result["input"] == long_text
        assert "[GUARDRAILED]" not in result["input"]


class TestInputProcessingListOfStrings:
    """Test input processing for list of strings"""

    @pytest.mark.asyncio
    async def test_process_list_of_strings(self):
        """Test processing a list of strings"""
        handler = OpenAIEmbeddingHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {
            "model": "text-embedding-ada-002",
            "input": [
                "First text",
                "Second text",
                "Third text",
            ],
        }

        result = await handler.process_input_messages(data, guardrail)

        assert len(result["input"]) == 3
        assert result["input"][0] == "First text [GUARDRAILED]"
        assert result["input"][1] == "Second text [GUARDRAILED]"
        assert result["input"][2] == "Third text [GUARDRAILED]"

    @pytest.mark.asyncio
    async def test_process_empty_list(self):
        """Test processing an empty list"""
        handler = OpenAIEmbeddingHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {"model": "text-embedding-ada-002", "input": []}

        result = await handler.process_input_messages(data, guardrail)

        assert result["input"] == []

    @pytest.mark.asyncio
    async def test_process_single_item_list(self):
        """Test processing a list with a single item"""
        handler = OpenAIEmbeddingHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {"model": "text-embedding-ada-002", "input": ["Only text"]}

        result = await handler.process_input_messages(data, guardrail)

        assert len(result["input"]) == 1
        assert result["input"][0] == "Only text [GUARDRAILED]"

    @pytest.mark.asyncio
    async def test_process_large_list_of_strings(self):
        """Test processing a large list of strings"""
        handler = OpenAIEmbeddingHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        inputs = [f"Text number {i}" for i in range(100)]

        data = {"model": "text-embedding-ada-002", "input": inputs}

        result = await handler.process_input_messages(data, guardrail)

        assert len(result["input"]) == 100
        for i in range(100):
            assert result["input"][i] == f"Text number {i} [GUARDRAILED]"


class TestInputProcessingLengthLimits:
    """Test input processing with length limit scenarios"""

    @pytest.mark.asyncio
    @patch(
        "litellm.llms.openai.embedding.guardrail_translation.handler.MAX_GUARDRAIL_INPUT_LENGTH",
        20,
    )
    @patch(
        "litellm.llms.openai.embedding.guardrail_translation.handler.DISABLE_MAX_GUARDRAIL_INPUT_CHECK",
        False,
    )
    async def test_process_list_with_some_exceeding_limit(self):
        """Test processing list where some items exceed length limit"""
        handler = OpenAIEmbeddingHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {
            "model": "text-embedding-ada-002",
            "input": [
                "Short text",  # Under limit - should be guardrailed
                "This is a very long text that exceeds the limit",  # Over limit - skip
                "Another short",  # Under limit - should be guardrailed
                "Yet another extremely long text that definitely exceeds limit",  # Over limit - skip
                "Last short",  # Under limit - should be guardrailed
            ],
        }

        result = await handler.process_input_messages(data, guardrail)

        # Check that order is preserved and correct items are guardrailed
        assert len(result["input"]) == 5
        assert result["input"][0] == "Short text [GUARDRAILED]"
        assert result["input"][1] == "This is a very long text that exceeds the limit"
        assert result["input"][2] == "Another short [GUARDRAILED]"
        assert (
            result["input"][3]
            == "Yet another extremely long text that definitely exceeds limit"
        )
        assert result["input"][4] == "Last short [GUARDRAILED]"

    @pytest.mark.asyncio
    @patch(
        "litellm.llms.openai.embedding.guardrail_translation.handler.MAX_GUARDRAIL_INPUT_LENGTH",
        10,
    )
    @patch(
        "litellm.llms.openai.embedding.guardrail_translation.handler.DISABLE_MAX_GUARDRAIL_INPUT_CHECK",
        False,
    )
    async def test_process_list_all_exceeding_limit(self):
        """Test processing list where all items exceed length limit"""
        handler = OpenAIEmbeddingHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {
            "model": "text-embedding-ada-002",
            "input": [
                "This is long text one",
                "This is long text two",
                "This is long text three",
            ],
        }

        result = await handler.process_input_messages(data, guardrail)

        # All should be unchanged (not guardrailed)
        assert len(result["input"]) == 3
        assert result["input"][0] == "This is long text one"
        assert result["input"][1] == "This is long text two"
        assert result["input"][2] == "This is long text three"

    @pytest.mark.asyncio
    @patch(
        "litellm.llms.openai.embedding.guardrail_translation.handler.MAX_GUARDRAIL_INPUT_LENGTH",
        1000,
    )
    @patch(
        "litellm.llms.openai.embedding.guardrail_translation.handler.DISABLE_MAX_GUARDRAIL_INPUT_CHECK",
        False,
    )
    async def test_process_list_none_exceeding_limit(self):
        """Test processing list where no items exceed length limit"""
        handler = OpenAIEmbeddingHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {
            "model": "text-embedding-ada-002",
            "input": [
                "Short one",
                "Short two",
                "Short three",
            ],
        }

        result = await handler.process_input_messages(data, guardrail)

        # All should be guardrailed
        assert len(result["input"]) == 3
        assert result["input"][0] == "Short one [GUARDRAILED]"
        assert result["input"][1] == "Short two [GUARDRAILED]"
        assert result["input"][2] == "Short three [GUARDRAILED]"

    @pytest.mark.asyncio
    @patch(
        "litellm.llms.openai.embedding.guardrail_translation.handler.MAX_GUARDRAIL_INPUT_LENGTH",
        10,
    )
    @patch(
        "litellm.llms.openai.embedding.guardrail_translation.handler.DISABLE_MAX_GUARDRAIL_INPUT_CHECK",
        True,
    )
    async def test_process_with_disabled_length_check(self):
        """Test that length check can be disabled"""
        handler = OpenAIEmbeddingHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {
            "model": "text-embedding-ada-002",
            "input": ["This is a very long text that would normally exceed the limit"],
        }

        result = await handler.process_input_messages(data, guardrail)

        # Should be guardrailed even though it's long (check is disabled)
        assert (
            result["input"][0]
            == "This is a very long text that would normally exceed the limit [GUARDRAILED]"
        )


class TestInputProcessingTokenArrays:
    """Test input processing for token arrays (should be skipped)"""

    @pytest.mark.asyncio
    async def test_process_token_arrays(self):
        """Test that token arrays are not processed"""
        handler = OpenAIEmbeddingHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {
            "model": "text-embedding-ada-002",
            "input": [[1, 2, 3, 4], [5, 6, 7, 8]],
        }

        result = await handler.process_input_messages(data, guardrail)

        # Should return unchanged (can't guardrail token arrays)
        assert result["input"] == [[1, 2, 3, 4], [5, 6, 7, 8]]

    @pytest.mark.asyncio
    async def test_process_mixed_input_types(self):
        """Test that mixed input types are skipped"""
        handler = OpenAIEmbeddingHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        # Mix of strings and lists (invalid, but should handle gracefully)
        data = {
            "model": "text-embedding-ada-002",
            "input": ["text", [1, 2, 3]],
        }

        result = await handler.process_input_messages(data, guardrail)

        # Should return unchanged (mixed types not supported)
        assert result["input"] == ["text", [1, 2, 3]]


class TestOutputProcessing:
    """Test output processing functionality"""

    @pytest.mark.asyncio
    async def test_output_processing_returns_unchanged(self):
        """Test that output processing returns response unchanged"""
        handler = OpenAIEmbeddingHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        # Create a mock EmbeddingResponse
        response = EmbeddingResponse(
            model="text-embedding-ada-002",
            data=[
                {
                    "embedding": [0.1, 0.2, 0.3],
                    "index": 0,
                    "object": "embedding",
                }
            ],
            object="list",
            usage={"prompt_tokens": 10, "total_tokens": 10},
        )

        result = await handler.process_output_response(response, guardrail)

        # Should return unchanged (embeddings don't contain text)
        assert result == response
        assert result.model == "text-embedding-ada-002"


class TestPIIMaskingScenario:
    """Test real-world scenario: PII masking in embedding inputs"""

    @pytest.mark.asyncio
    async def test_pii_masking_single_text(self):
        """Test that PII can be masked from embedding inputs"""

        class PIIMaskingGuardrail(CustomGuardrail):
            """Mock PII masking guardrail"""

            async def apply_guardrail(self, text: str) -> str:
                import re

                masked = re.sub(
                    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                    "[EMAIL_REDACTED]",
                    text,
                )
                masked = masked.replace("John Doe", "[NAME_REDACTED]")
                return masked

        handler = OpenAIEmbeddingHandler()
        guardrail = PIIMaskingGuardrail(guardrail_name="mask_pii")

        data = {
            "model": "text-embedding-ada-002",
            "input": "Contact John Doe at john@example.com for more information",
        }

        result = await handler.process_input_messages(data, guardrail)

        # Verify PII was masked
        assert "john@example.com" not in result["input"]
        assert "John Doe" not in result["input"]
        assert "[EMAIL_REDACTED]" in result["input"]
        assert "[NAME_REDACTED]" in result["input"]

    @pytest.mark.asyncio
    async def test_pii_masking_multiple_texts(self):
        """Test PII masking across multiple texts"""

        class PIIMaskingGuardrail(CustomGuardrail):
            """Mock PII masking guardrail"""

            async def apply_guardrail(self, text: str) -> str:
                import re

                # Mask phone numbers
                masked = re.sub(r"\d{3}-\d{3}-\d{4}", "[PHONE_REDACTED]", text)
                # Mask emails
                masked = re.sub(
                    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                    "[EMAIL_REDACTED]",
                    masked,
                )
                return masked

        handler = OpenAIEmbeddingHandler()
        guardrail = PIIMaskingGuardrail(guardrail_name="mask_pii")

        data = {
            "model": "text-embedding-ada-002",
            "input": [
                "Call me at 555-123-4567",
                "Email support@company.com",
                "My number is 555-987-6543 and email is user@domain.com",
            ],
        }

        result = await handler.process_input_messages(data, guardrail)

        # Verify PII was masked in all texts
        assert "555-123-4567" not in result["input"][0]
        assert "[PHONE_REDACTED]" in result["input"][0]

        assert "support@company.com" not in result["input"][1]
        assert "[EMAIL_REDACTED]" in result["input"][1]

        assert "555-987-6543" not in result["input"][2]
        assert "user@domain.com" not in result["input"][2]
        assert result["input"][2].count("[PHONE_REDACTED]") == 1
        assert result["input"][2].count("[EMAIL_REDACTED]") == 1


class TestContentModerationScenario:
    """Test real-world scenario: Content moderation"""

    @pytest.mark.asyncio
    async def test_profanity_filtering(self):
        """Test filtering profanity from embedding inputs"""

        class ProfanityFilterGuardrail(CustomGuardrail):
            """Mock profanity filter guardrail"""

            async def apply_guardrail(self, text: str) -> str:
                bad_words = ["badword", "inappropriate", "offensive"]
                filtered = text
                for word in bad_words:
                    filtered = filtered.replace(word, "[FILTERED]")
                return filtered

        handler = OpenAIEmbeddingHandler()
        guardrail = ProfanityFilterGuardrail(guardrail_name="content_filter")

        data = {
            "model": "text-embedding-ada-002",
            "input": [
                "This is a clean sentence",
                "This contains badword content",
                "Some inappropriate and offensive text",
            ],
        }

        result = await handler.process_input_messages(data, guardrail)

        # First text should be unchanged (no bad words)
        assert result["input"][0] == "This is a clean sentence"

        # Other texts should have filtered words
        assert "badword" not in result["input"][1]
        assert "[FILTERED]" in result["input"][1]

        assert "inappropriate" not in result["input"][2]
        assert "offensive" not in result["input"][2]
        assert result["input"][2].count("[FILTERED]") == 2


class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.mark.asyncio
    async def test_process_with_none_input(self):
        """Test processing with None as input value"""
        handler = OpenAIEmbeddingHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {"model": "text-embedding-ada-002", "input": None}

        result = await handler.process_input_messages(data, guardrail)

        # Should return unchanged when input is None
        assert result["input"] is None

    @pytest.mark.asyncio
    async def test_process_preserves_other_params(self):
        """Test that other parameters are preserved"""
        handler = OpenAIEmbeddingHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {
            "model": "text-embedding-ada-002",
            "input": "Test text",
            "encoding_format": "float",
            "dimensions": 256,
            "user": "test_user",
        }

        result = await handler.process_input_messages(data, guardrail)

        # Input should be modified
        assert result["input"] == "Test text [GUARDRAILED]"

        # Other params should be preserved
        assert result["model"] == "text-embedding-ada-002"
        assert result["encoding_format"] == "float"
        assert result["dimensions"] == 256
        assert result["user"] == "test_user"

    @pytest.mark.asyncio
    async def test_reconstruction_order_with_alternating_lengths(self):
        """Test that reconstruction maintains order with alternating short/long texts"""

        class IndexTrackingGuardrail(CustomGuardrail):
            """Guardrail that includes index in output for verification"""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.call_count = 0

            async def apply_guardrail(self, text: str) -> str:
                result = f"{text} [CALL_{self.call_count}]"
                self.call_count += 1
                return result

        handler = OpenAIEmbeddingHandler()
        guardrail = IndexTrackingGuardrail(guardrail_name="test")

        with patch(
            "litellm.llms.openai.embedding.guardrail_translation.handler.MAX_GUARDRAIL_INPUT_LENGTH",
            20,
        ):
            with patch(
                "litellm.llms.openai.embedding.guardrail_translation.handler.DISABLE_MAX_GUARDRAIL_INPUT_CHECK",
                False,
            ):
                data = {
                    "model": "text-embedding-ada-002",
                    "input": [
                        "Short 0",  # Under limit
                        "This is a very long text that exceeds limit 1",  # Over limit
                        "Short 2",  # Under limit
                        "Short 3",  # Under limit
                        "Another very long text that exceeds the limit 4",  # Over limit
                        "Short 5",  # Under limit
                    ],
                }

                result = await handler.process_input_messages(data, guardrail)

                # Verify order and correct processing
                assert result["input"][0] == "Short 0 [CALL_0]"
                assert (
                    result["input"][1]
                    == "This is a very long text that exceeds limit 1"
                )
                assert result["input"][2] == "Short 2 [CALL_1]"
                assert result["input"][3] == "Short 3 [CALL_2]"
                assert (
                    result["input"][4]
                    == "Another very long text that exceeds the limit 4"
                )
                assert result["input"][5] == "Short 5 [CALL_3]"
