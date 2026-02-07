"""OpenAI Audio Transcription handler for Unified Guardrails."""

from litellm.llms.openai.transcriptions.guardrail_translation.handler import (
    OpenAIAudioTranscriptionHandler,
)
from litellm.types.utils import CallTypes

guardrail_translation_mappings = {
    CallTypes.transcription: OpenAIAudioTranscriptionHandler,
    CallTypes.atranscription: OpenAIAudioTranscriptionHandler,
}

__all__ = ["guardrail_translation_mappings", "OpenAIAudioTranscriptionHandler"]
