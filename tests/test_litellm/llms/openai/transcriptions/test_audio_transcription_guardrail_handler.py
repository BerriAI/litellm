"""
Unit tests for OpenAI Audio Transcription Guardrail Translation Handler
"""

import os
import sys
from typing import List, Optional, Tuple

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms import get_guardrail_translation_mapping
from litellm.llms.openai.transcriptions.guardrail_translation.handler import (
    OpenAIAudioTranscriptionHandler,
)
from litellm.types.utils import CallTypes
from litellm.utils import TranscriptionResponse


class MockGuardrail(CustomGuardrail):
    """Mock guardrail for testing"""

    async def apply_guardrail(
        self, inputs: dict, request_data: dict, input_type: str, **kwargs
    ) -> dict:
        texts = inputs.get("texts", [])
        return {"texts": [f"{text} [GUARDRAILED]" for text in texts]}


class TestHandlerDiscovery:
    """Test that the handler is properly discovered"""

    def test_handler_discovered_for_transcription(self):
        """Test that transcription CallType is mapped to handler"""
        handler_class = get_guardrail_translation_mapping(CallTypes.transcription)
        assert handler_class == OpenAIAudioTranscriptionHandler

    def test_handler_discovered_for_atranscription(self):
        """Test that atranscription CallType is mapped to handler"""
        handler_class = get_guardrail_translation_mapping(CallTypes.atranscription)
        assert handler_class == OpenAIAudioTranscriptionHandler


class TestInputProcessing:
    """Test input processing functionality"""

    @pytest.mark.asyncio
    async def test_input_processing_is_noop(self):
        """Test that input processing returns data unchanged (audio files can't be guardrailed)"""
        handler = OpenAIAudioTranscriptionHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {
            "model": "whisper-1",
            "file": "<audio file object>",
            "response_format": "json",
        }

        result = await handler.process_input_messages(data, guardrail)

        # Should return unchanged since audio files can't be text-guardrailed
        assert result == data
        assert result["model"] == "whisper-1"
        assert result["file"] == "<audio file object>"


class TestOutputProcessing:
    """Test output processing functionality"""

    @pytest.mark.asyncio
    async def test_process_simple_transcription(self):
        """Test processing a simple transcription response"""
        handler = OpenAIAudioTranscriptionHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        response = TranscriptionResponse(
            text="This is a test transcription",
        )

        result = await handler.process_output_response(response, guardrail)

        assert result.text == "This is a test transcription [GUARDRAILED]"

    @pytest.mark.asyncio
    async def test_process_transcription_with_metadata(self):
        """Test processing transcription with additional metadata"""
        handler = OpenAIAudioTranscriptionHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        response = TranscriptionResponse(
            text="Hello world",
        )
        # Add metadata
        response.duration = 3.5
        response.language = "en"

        result = await handler.process_output_response(response, guardrail)

        # Text should be guardrailed
        assert result.text == "Hello world [GUARDRAILED]"
        # Metadata should be preserved
        assert result.duration == 3.5
        assert result.language == "en"

    @pytest.mark.asyncio
    async def test_process_empty_text(self):
        """Test processing transcription with None text"""
        handler = OpenAIAudioTranscriptionHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        response = TranscriptionResponse(text=None)

        result = await handler.process_output_response(response, guardrail)

        # Should return unchanged when text is None
        assert result.text is None

    @pytest.mark.asyncio
    async def test_process_long_transcription(self):
        """Test processing a long transcription"""
        handler = OpenAIAudioTranscriptionHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        long_text = "This is a very long transcription. " * 100

        response = TranscriptionResponse(text=long_text)

        result = await handler.process_output_response(response, guardrail)

        assert result.text == f"{long_text} [GUARDRAILED]"
        assert long_text in result.text


class TestPIIMaskingScenario:
    """Test real-world scenario: PII masking in transcriptions"""

    @pytest.mark.asyncio
    async def test_pii_masking_in_transcription(self):
        """Test that PII can be masked from transcribed audio"""

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
                    masked = masked.replace("555-1234", "[PHONE_REDACTED]")
                    masked_texts.append(masked)
                return {"texts": masked_texts}

        handler = OpenAIAudioTranscriptionHandler()
        guardrail = PIIMaskingGuardrail(guardrail_name="mask_pii")

        # Simulate transcription that contains PII
        response = TranscriptionResponse(
            text="My name is John Doe and you can reach me at john@example.com or call 555-1234"
        )

        result = await handler.process_output_response(response, guardrail)

        # Verify PII was masked
        assert "john@example.com" not in result.text
        assert "John Doe" not in result.text
        assert "555-1234" not in result.text
        assert "[EMAIL_REDACTED]" in result.text
        assert "[NAME_REDACTED]" in result.text
        assert "[PHONE_REDACTED]" in result.text

    @pytest.mark.asyncio
    async def test_meeting_transcription_pii_redaction(self):
        """Test PII redaction in a meeting transcription scenario"""

        class PIIMaskingGuardrail(CustomGuardrail):
            """Mock PII masking guardrail"""

            async def apply_guardrail(
                self, inputs: dict, request_data: dict, input_type: str, **kwargs
            ) -> dict:
                import re

                texts = inputs.get("texts", [])
                masked_texts = []
                for text in texts:
                    # Mask credit card numbers
                    masked = re.sub(
                        r"\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}", "[CC_REDACTED]", text
                    )
                    # Mask SSNs
                    masked = re.sub(r"\d{3}-\d{2}-\d{4}", "[SSN_REDACTED]", masked)
                    # Mask emails
                    masked = re.sub(
                        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                        "[EMAIL_REDACTED]",
                        masked,
                    )
                    masked_texts.append(masked)
                return {"texts": masked_texts}

        handler = OpenAIAudioTranscriptionHandler()
        guardrail = PIIMaskingGuardrail(guardrail_name="mask_pii")

        # Simulate a meeting transcription with sensitive data
        meeting_text = (
            "During our call, the customer mentioned their card number is 4532-1234-5678-9010 "
            "and their SSN is 123-45-6789. You can email them at customer@company.com for follow-up."
        )

        response = TranscriptionResponse(text=meeting_text)
        result = await handler.process_output_response(response, guardrail)

        # Verify all PII was masked
        assert "4532-1234-5678-9010" not in result.text
        assert "123-45-6789" not in result.text
        assert "customer@company.com" not in result.text
        assert "[CC_REDACTED]" in result.text
        assert "[SSN_REDACTED]" in result.text
        assert "[EMAIL_REDACTED]" in result.text


class TestContentModerationScenario:
    """Test real-world scenario: Content moderation on transcriptions"""

    @pytest.mark.asyncio
    async def test_profanity_filtering(self):
        """Test filtering profanity from transcriptions"""

        class ProfanityFilterGuardrail(CustomGuardrail):
            """Mock profanity filter guardrail"""

            async def apply_guardrail(
                self, inputs: dict, request_data: dict, input_type: str, **kwargs
            ) -> dict:
                # Simple mock: replace common profanity
                bad_words = ["badword1", "badword2", "inappropriate"]
                texts = inputs.get("texts", [])
                filtered_texts = []
                for text in texts:
                    filtered = text
                    for word in bad_words:
                        filtered = filtered.replace(word, "[FILTERED]")
                    filtered_texts.append(filtered)
                return {"texts": filtered_texts}

        handler = OpenAIAudioTranscriptionHandler()
        guardrail = ProfanityFilterGuardrail(guardrail_name="content_filter")

        response = TranscriptionResponse(
            text="This contains badword1 and inappropriate content with badword2"
        )

        result = await handler.process_output_response(response, guardrail)

        # Verify profanity was filtered
        assert "badword1" not in result.text
        assert "badword2" not in result.text
        assert "inappropriate" not in result.text
        assert result.text.count("[FILTERED]") == 3


class TestMultilingualTranscription:
    """Test scenarios with different languages"""

    @pytest.mark.asyncio
    async def test_transcription_preserves_language_info(self):
        """Test that language metadata is preserved when applying guardrails"""
        handler = OpenAIAudioTranscriptionHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        response = TranscriptionResponse(text="Bonjour le monde")
        response.language = "fr"

        result = await handler.process_output_response(response, guardrail)

        # Text should be guardrailed
        assert result.text == "Bonjour le monde [GUARDRAILED]"
        # Language metadata should be preserved
        assert result.language == "fr"
