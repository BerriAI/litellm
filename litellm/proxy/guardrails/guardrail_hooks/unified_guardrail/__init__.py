from litellm.llms.openai.chat.guardrail_translation.handler import (
    OpenAIChatCompletionsHandler,
)
from litellm.types.utils import CallTypes

endpoint_translation_mappings = {
    CallTypes.completion: OpenAIChatCompletionsHandler,
    CallTypes.acompletion: OpenAIChatCompletionsHandler,
}

__all__ = ["endpoint_translation_mappings"]
