"""
Translates from OpenAI's `/v1/chat/completions` to Together AI's `/v1/chat/completions`.

Docs: https://docs.together.ai/docs/chat-overview
"""

from types import MappingProxyType
from typing import Final

from litellm._logging import verbose_logger
from litellm.utils import supports_function_calling

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

FUNCTION_CALLING_ONLY_PARAMS: Final = ("tools", "tool_choice", "function_call", "response_format")
PLAIN_TEXT_RESPONSE_FORMAT: Final = MappingProxyType({"type": "text"})


class TogetherAIChatConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> list:
        supports_fc: bool | None = None
        try:
            supports_fc = supports_function_calling(model, custom_llm_provider="together_ai")
        except Exception as e:
            verbose_logger.debug("Error getting supported openai params: %s", e)

        supported_params: Final = super().get_supported_openai_params(model)
        if supports_fc is True:
            return supported_params
        verbose_logger.debug(
            "Only some together models support function calling/response_format. Docs - https://docs.together.ai/docs/function-calling"
        )
        return [  # mutable-ok: the inherited contract returns a plain list; building fresh avoids mutating the base class's value
            param for param in supported_params if param not in FUNCTION_CALLING_ONLY_PARAMS
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        mapped_openai_params: Final = super().map_openai_params(non_default_params, optional_params, model, drop_params)

        if mapped_openai_params.get("response_format") == PLAIN_TEXT_RESPONSE_FORMAT:
            mapped_openai_params.pop("response_format")
        return mapped_openai_params
