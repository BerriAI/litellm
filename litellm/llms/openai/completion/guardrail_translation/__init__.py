"""OpenAI Text Completion handler for Unified Guardrails."""

from litellm.llms.openai.completion.guardrail_translation.handler import (
    OpenAITextCompletionHandler,
)
from litellm.types.utils import CallTypes

guardrail_translation_mappings = {
    CallTypes.text_completion: OpenAITextCompletionHandler,
    CallTypes.atext_completion: OpenAITextCompletionHandler,
}

__all__ = ["OpenAITextCompletionHandler", "guardrail_translation_mappings"]
