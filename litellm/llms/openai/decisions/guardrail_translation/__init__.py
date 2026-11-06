from typing import Final

from litellm.llms.openai.decisions.guardrail_translation.handler import DecisionsHandler, SystemOneHandler
from litellm.types.utils import CallTypes

guardrail_translation_mappings: Final = {
    CallTypes.decisions: DecisionsHandler,
    CallTypes.adecisions: DecisionsHandler,
    CallTypes.systemone: SystemOneHandler,
    CallTypes.asystemone: SystemOneHandler,
}

__all__ = ["DecisionsHandler", "SystemOneHandler", "guardrail_translation_mappings"]
