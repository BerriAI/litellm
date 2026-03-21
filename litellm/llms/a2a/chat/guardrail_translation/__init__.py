"""A2A Protocol handler for Unified Guardrails."""

from litellm.llms.a2a.chat.guardrail_translation.handler import A2AGuardrailHandler
from litellm.types.utils import CallTypes

guardrail_translation_mappings = {
    CallTypes.send_message: A2AGuardrailHandler,
    CallTypes.asend_message: A2AGuardrailHandler,
}

__all__ = ["guardrail_translation_mappings"]
