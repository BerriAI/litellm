"""
Unit tests for OpenAI Image Generation Guardrail Translation Handler
"""

import os
import sys
from typing import List, Optional, Tuple

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.llms import get_guardrail_translation_mapping
from litellm.llms.openai.image_generation.guardrail_translation.handler import (
    OpenAIImageGenerationHandler,
)
from litellm.types.utils import CallTypes, ImageObject, ImageResponse


class MockGuardrail(CustomGuardrail):
    """Mock guardrail for testing"""

    async def apply_guardrail(
        self, inputs: dict, request_data: dict, input_type: str, **kwargs
    ) -> dict:
        texts = inputs.get("texts", [])
        return {"texts": [f"{text} [GUARDRAILED]" for text in texts]}


class TestHandlerDiscovery:
    """Test that the handler is properly discovered"""

    def test_handler_discovered_for_image_generation(self):
        """Test that image_generation CallType is mapped to handler"""
        handler_class = get_guardrail_translation_mapping(CallTypes.image_generation)
        assert handler_class == OpenAIImageGenerationHandler

    def test_handler_discovered_for_aimage_generation(self):
        """Test that aimage_generation CallType is mapped to handler"""
        handler_class = get_guardrail_translation_mapping(CallTypes.aimage_generation)
        assert handler_class == OpenAIImageGenerationHandler


class TestInputProcessing:
    """Test input processing functionality"""

    @pytest.mark.asyncio
    async def test_process_simple_prompt(self):
        """Test processing a simple text prompt"""
        handler = OpenAIImageGenerationHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {
            "model": "dall-e-3",
            "prompt": "A cute baby sea otter",
            "size": "1024x1024",
        }

        result = await handler.process_input_messages(data, guardrail)

        assert result["prompt"] == "A cute baby sea otter [GUARDRAILED]"
        assert result["model"] == "dall-e-3"
        assert result["size"] == "1024x1024"

    @pytest.mark.asyncio
    async def test_process_no_prompt(self):
        """Test processing when no prompt is provided"""
        handler = OpenAIImageGenerationHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {"model": "dall-e-3", "size": "1024x1024"}

        result = await handler.process_input_messages(data, guardrail)

        # Should return data unchanged when no prompt
        assert result == data
        assert "prompt" not in result

    @pytest.mark.asyncio
    async def test_process_empty_prompt(self):
        """Test processing when prompt is None"""
        handler = OpenAIImageGenerationHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        data = {"model": "dall-e-3", "prompt": None}

        result = await handler.process_input_messages(data, guardrail)

        # Should return data unchanged when prompt is None
        assert result["prompt"] is None

    @pytest.mark.asyncio
    async def test_process_complex_prompt(self):
        """Test processing a complex/long prompt"""
        handler = OpenAIImageGenerationHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        long_prompt = (
            "A highly detailed digital painting of a futuristic cityscape at sunset, "
            "with flying cars, neon signs, and towering skyscrapers reflecting in the water, "
            "in the style of Blade Runner, 8k resolution, photorealistic"
        )

        data = {"model": "dall-e-3", "prompt": long_prompt}

        result = await handler.process_input_messages(data, guardrail)

        assert result["prompt"] == f"{long_prompt} [GUARDRAILED]"
        assert long_prompt in result["prompt"]


class TestOutputProcessing:
    """Test output processing functionality"""

    @pytest.mark.asyncio
    async def test_process_output_returns_unchanged(self):
        """Test that output processing returns response unchanged"""
        handler = OpenAIImageGenerationHandler()
        guardrail = MockGuardrail(guardrail_name="test")

        # Create a mock ImageResponse
        image_object = ImageObject(
            url="https://example.com/image.png",
            revised_prompt="A cute baby sea otter",
        )
        response = ImageResponse(created=1589478378, data=[image_object])

        result = await handler.process_output_response(response, guardrail)

        # Should return unchanged
        assert result == response
        assert result.created == 1589478378
        assert len(result.data) == 1


class TestPIIMaskingScenario:
    """Test real-world scenario: PII masking in prompts"""

    @pytest.mark.asyncio
    async def test_pii_masking_in_prompt(self):
        """Test that PII can be masked from image generation prompts"""

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

        handler = OpenAIImageGenerationHandler()
        guardrail = PIIMaskingGuardrail(guardrail_name="mask_pii")

        data = {
            "model": "dall-e-3",
            "prompt": "Generate an image of John Doe at john@example.com",
        }

        result = await handler.process_input_messages(data, guardrail)

        # Verify PII was masked
        assert "john@example.com" not in result["prompt"]
        assert "John Doe" not in result["prompt"]
        assert "[EMAIL_REDACTED]" in result["prompt"]
        assert "[NAME_REDACTED]" in result["prompt"]
