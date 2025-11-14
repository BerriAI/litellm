"""OpenAI Text-to-Speech handler for Unified Guardrails."""

from litellm.llms.openai.speech.guardrail_translation.handler import (
    OpenAITextToSpeechHandler,
)
from litellm.types.utils import CallTypes

guardrail_translation_mappings = {
    CallTypes.speech: OpenAITextToSpeechHandler,
    CallTypes.aspeech: OpenAITextToSpeechHandler,
}

__all__ = ["guardrail_translation_mappings", "OpenAITextToSpeechHandler"]
