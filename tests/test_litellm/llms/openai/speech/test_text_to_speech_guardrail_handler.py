"""
Unit tests for OpenAI Text-to-Speech Guardrail Translation Handler
"""

import os
import sys
from typing import List, Optional, Tuple

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms import get_guardrail_translation_mapping
from litellm.llms.openai.speech.guardrail_translation.handler import (
    OpenAITextToSpeechHandler,
)
from litellm.types.utils import CallTypes


class MockGuardrail(CustomGuardrail):
    """Mock guardrail for testing"""

    async def apply_guardrail(
        self, inputs: dict, request_data: dict, input_type: str, **kwargs
    ) -> dict:
        texts = inputs.get("texts", [])
        return {"texts": [f"{text} [GUARDRAILED]" for text in texts]}


class MockBinaryResponse:
    """Mock binary response for audio output"""

    def __init__(self, content: bytes = b"fake audio data"):
        self.content = content
        self.response = self

    def stream_to_file(self, path):
        with open(path, "wb") as f:
            f.write(self.content)


class TestHandlerDiscovery:
    """Test that the handler is properly discovered"""

    def test_handler_discovered_for_speech(self):
        """Test that speech CallType is mapped to handler"""
        handler_class = get_guardrail_translation_mapping(CallTypes.speech)
        assert handler_class == OpenAITextToSpeechHandler

    def test_handler_discovered_for_aspeech(self):
        """Test that aspeech CallType is mapped to handler"""
        handler_class = get_guardrail_translation_mapping(CallTypes.aspeech)
        assert handler_class == OpenAITextToSpeechHandler


class TestInputProcessing:
    """Test input processing functionality"""

    @pytest.mark.asyncio
    async def test_process_simple_input_text(self):
        """Test processing a simple input text"""
        handler = OpenAITextToSpeechHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {
            "model": "tts-1",
            "input": "The quick brown fox jumped over the lazy dog.",
            "voice": "alloy",
        }

        result = await handler.process_input_messages(data, guardrail)

        assert (
            result["input"]
            == "The quick brown fox jumped over the lazy dog. [GUARDRAILED]"
        )
        assert result["model"] == "tts-1"
        assert result["voice"] == "alloy"

    @pytest.mark.asyncio
    async def test_process_no_input(self):
        """Test processing when no input is provided"""
        handler = OpenAITextToSpeechHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {"model": "tts-1", "voice": "alloy"}

        result = await handler.process_input_messages(data, guardrail)

        # Should return data unchanged when no input
        assert result == data
        assert "input" not in result

    @pytest.mark.asyncio
    async def test_process_empty_input(self):
        """Test processing when input is None"""
        handler = OpenAITextToSpeechHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {"model": "tts-1", "input": None, "voice": "alloy"}

        result = await handler.process_input_messages(data, guardrail)

        # Should return data unchanged when input is None
        assert result["input"] is None

    @pytest.mark.asyncio
    async def test_process_long_input(self):
        """Test processing a long input text"""
        handler = OpenAITextToSpeechHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        long_text = "This is a very long text that will be converted to speech. " * 50

        data = {"model": "tts-1-hd", "input": long_text, "voice": "shimmer"}

        result = await handler.process_input_messages(data, guardrail)

        assert result["input"] == f"{long_text} [GUARDRAILED]"
        assert long_text in result["input"]

    @pytest.mark.asyncio
    async def test_process_with_all_optional_params(self):
        """Test processing with all optional TTS parameters"""
        handler = OpenAITextToSpeechHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {
            "model": "tts-1",
            "input": "Hello world",
            "voice": "nova",
            "response_format": "wav",
            "speed": 1.5,
        }

        result = await handler.process_input_messages(data, guardrail)

        # Input should be guardrailed
        assert result["input"] == "Hello world [GUARDRAILED]"
        # Other params should be preserved
        assert result["voice"] == "nova"
        assert result["response_format"] == "wav"
        assert result["speed"] == 1.5


class TestOutputProcessing:
    """Test output processing functionality"""

    @pytest.mark.asyncio
    async def test_output_processing_is_noop(self):
        """Test that output processing returns response unchanged (audio is binary)"""
        handler = OpenAITextToSpeechHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        response = MockBinaryResponse(content=b"fake audio mp3 data")

        result = await handler.process_output_response(response, guardrail)

        # Should return unchanged since audio can't be text-guardrailed
        assert result == response
        assert result.content == b"fake audio mp3 data"


class TestPIIMaskingScenario:
    """Test real-world scenario: PII masking in TTS input"""

    @pytest.mark.asyncio
    async def test_pii_masking_in_tts_input(self):
        """Test that PII can be masked from text before TTS conversion"""

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

        handler = OpenAITextToSpeechHandler()
        guardrail = PIIMaskingGuardrail(guardrail_name="mask_pii")

        data = {
            "model": "tts-1",
            "input": "Hi, I'm John Doe. You can reach me at john@example.com or call 555-1234",
            "voice": "alloy",
        }

        result = await handler.process_input_messages(data, guardrail)

        # Verify PII was masked
        assert "john@example.com" not in result["input"]
        assert "John Doe" not in result["input"]
        assert "555-1234" not in result["input"]
        assert "[EMAIL_REDACTED]" in result["input"]
        assert "[NAME_REDACTED]" in result["input"]
        assert "[PHONE_REDACTED]" in result["input"]

    @pytest.mark.asyncio
    async def test_phone_message_pii_redaction(self):
        """Test PII redaction in an automated phone message scenario"""

        class PIIMaskingGuardrail(CustomGuardrail):
            """Mock PII masking guardrail"""

            async def apply_guardrail(
                self, inputs: dict, request_data: dict, input_type: str, **kwargs
            ) -> dict:
                import re

                texts = inputs.get("texts", [])
                masked_texts = []
                for text in texts:
                    # Mask account numbers
                    masked = re.sub(
                        r"account number \d{8,12}", "account number [REDACTED]", text
                    )
                    # Mask SSNs
                    masked = re.sub(r"\d{3}-\d{2}-\d{4}", "[SSN_REDACTED]", masked)
                    # Mask credit cards
                    masked = re.sub(
                        r"\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}", "[CC_REDACTED]", masked
                    )
                    masked_texts.append(masked)
                return {"texts": masked_texts}

        handler = OpenAITextToSpeechHandler()
        guardrail = PIIMaskingGuardrail(guardrail_name="mask_pii")

        # Simulate an automated phone system message
        phone_message = (
            "Your account number 123456789 has a balance of $500. "
            "Please verify your SSN 123-45-6789 and card ending in 4532 1234 5678 9010."
        )

        data = {"model": "tts-1", "input": phone_message, "voice": "onyx"}

        result = await handler.process_input_messages(data, guardrail)

        # Verify all sensitive data was masked
        assert "123456789" not in result["input"]
        assert "123-45-6789" not in result["input"]
        assert "4532 1234 5678 9010" not in result["input"]
        assert "[REDACTED]" in result["input"]
        assert "[SSN_REDACTED]" in result["input"]
        assert "[CC_REDACTED]" in result["input"]


class TestContentModerationScenario:
    """Test real-world scenario: Content moderation for TTS"""

    @pytest.mark.asyncio
    async def test_profanity_filtering_before_tts(self):
        """Test filtering inappropriate content before TTS"""

        class ContentFilterGuardrail(CustomGuardrail):
            """Mock content filter guardrail"""

            async def apply_guardrail(
                self, inputs: dict, request_data: dict, input_type: str, **kwargs
            ) -> dict:
                # Simple mock: filter inappropriate words
                bad_words = ["badword", "inappropriate", "offensive"]
                texts = inputs.get("texts", [])
                filtered_texts = []
                for text in texts:
                    filtered = text
                    for word in bad_words:
                        filtered = filtered.replace(word, "[FILTERED]")
                    filtered_texts.append(filtered)
                return {"texts": filtered_texts}

        handler = OpenAITextToSpeechHandler()
        guardrail = ContentFilterGuardrail(guardrail_name="content_filter")

        data = {
            "model": "tts-1",
            "input": "This message contains badword and inappropriate offensive content",
            "voice": "fable",
        }

        result = await handler.process_input_messages(data, guardrail)

        # Verify inappropriate content was filtered
        assert "badword" not in result["input"]
        assert "inappropriate" not in result["input"]
        assert "offensive" not in result["input"]
        assert result["input"].count("[FILTERED]") == 3


class TestMultilingualTTS:
    """Test scenarios with different languages and voices"""

    @pytest.mark.asyncio
    async def test_multilingual_input_preserved(self):
        """Test that non-English text is properly processed"""
        handler = OpenAITextToSpeechHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {
            "model": "tts-1",
            "input": "Bonjour, comment allez-vous?",
            "voice": "alloy",
        }

        result = await handler.process_input_messages(data, guardrail)

        # Text should be guardrailed
        assert result["input"] == "Bonjour, comment allez-vous? [GUARDRAILED]"

    @pytest.mark.asyncio
    async def test_all_voice_options(self):
        """Test that guardrails work with all voice options"""
        handler = OpenAITextToSpeechHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

        for voice in voices:
            data = {
                "model": "tts-1",
                "input": f"Testing with {voice} voice",
                "voice": voice,
            }

            result = await handler.process_input_messages(data, guardrail)

            assert f"Testing with {voice} voice [GUARDRAILED]" == result["input"]
            assert result["voice"] == voice
