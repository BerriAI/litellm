"""OpenAI Image Generation handler for Unified Guardrails."""

from litellm.llms.openai.image_generation.guardrail_translation.handler import (
    OpenAIImageGenerationHandler,
)
from litellm.types.utils import CallTypes

guardrail_translation_mappings = {
    CallTypes.image_generation: OpenAIImageGenerationHandler,
    CallTypes.aimage_generation: OpenAIImageGenerationHandler,
}

__all__ = ["guardrail_translation_mappings", "OpenAIImageGenerationHandler"]
