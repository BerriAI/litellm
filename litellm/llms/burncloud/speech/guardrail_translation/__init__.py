"""OpenAI Text-to-Speech handler for Unified Guardrails."""

from litellm.llms.burncloud.speech.guardrail_translation.handler import (
    BurnCloudTextToSpeechHandler,
)
from litellm.types.utils import CallTypes

guardrail_translation_mappings = {
    CallTypes.speech: BurnCloudTextToSpeechHandler,
    CallTypes.aspeech: BurnCloudTextToSpeechHandler,
}

__all__ = ["guardrail_translation_mappings", "BurnCloudTextToSpeechHandler"]
