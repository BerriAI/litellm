"""OpenAI Video Generation handler for Unified Guardrails."""

from typing import Final

from litellm.llms.openai.videos.guardrail_translation.handler import (
    OpenAIVideoGenerationHandler,
)
from litellm.types.utils import CallTypes

guardrail_translation_mappings: Final = {
    CallTypes.avideo_generation: OpenAIVideoGenerationHandler,
    CallTypes.create_video: OpenAIVideoGenerationHandler,
    CallTypes.acreate_video: OpenAIVideoGenerationHandler,
    CallTypes.video_remix: OpenAIVideoGenerationHandler,
    CallTypes.avideo_remix: OpenAIVideoGenerationHandler,
    CallTypes.video_edit: OpenAIVideoGenerationHandler,
    CallTypes.avideo_edit: OpenAIVideoGenerationHandler,
    CallTypes.video_extension: OpenAIVideoGenerationHandler,
    CallTypes.avideo_extension: OpenAIVideoGenerationHandler,
}

__all__ = ["OpenAIVideoGenerationHandler", "guardrail_translation_mappings"]
