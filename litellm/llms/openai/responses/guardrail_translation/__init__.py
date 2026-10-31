"""OpenAI Responses API handler for Unified Guardrails."""

from typing import Final

from litellm.llms.openai.responses.guardrail_translation.handler import (
    OpenAIResponsesHandler,
)
from litellm.types.utils import CallTypes

guardrail_translation_mappings: Final = {
    CallTypes.responses: OpenAIResponsesHandler,
    CallTypes.aresponses: OpenAIResponsesHandler,
    CallTypes.compact_responses: OpenAIResponsesHandler,
    CallTypes.acompact_responses: OpenAIResponsesHandler,
}
__all__ = ["guardrail_translation_mappings"]
