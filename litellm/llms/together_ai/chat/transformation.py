"""
Translates from OpenAI's `/v1/chat/completions` to Together AI's `/v1/chat/completions`.

Docs: https://docs.together.ai/docs/chat-overview
"""

from collections.abc import Container
from types import MappingProxyType
from typing import Final

import litellm
from litellm._logging import verbose_logger
from litellm.exceptions import UnsupportedParamsError
from litellm.utils import supports_function_calling

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

TOOL_CALLING_PARAMS: Final = ("tools", "tool_choice", "function_call")
PLAIN_TEXT_RESPONSE_FORMAT: Final = MappingProxyType({"type": "text"})
FUNCTION_CALLING_DOCS_URL: Final = "https://docs.together.ai/docs/function-calling"


def _function_calling_verdict(model: str) -> bool | None:
    try:
        if supports_function_calling(model, custom_llm_provider="together_ai"):
            return True
    except Exception as e:
        verbose_logger.debug("Error checking together_ai function calling support for %s: %s", model, e)
    registry_entry: Final = litellm.model_cost.get(f"together_ai/{model}")
    if isinstance(registry_entry, dict) and registry_entry.get("supports_function_calling") is False:
        return False
    return None


def _tool_params_to_drop(passed_params: Container[str], model: str, drop_params: bool) -> tuple[str, ...]:
    passed_tool_params: Final = tuple(param for param in TOOL_CALLING_PARAMS if param in passed_params)
    if not passed_tool_params:
        return ()
    verdict: Final = _function_calling_verdict(model)
    if verdict is True:
        return ()
    if verdict is None:
        verbose_logger.warning(
            "together_ai model %s has no function calling entry in the model registry; passing %s through for Together to validate. Docs - %s",
            model,
            ", ".join(passed_tool_params),
            FUNCTION_CALLING_DOCS_URL,
        )
        return ()
    if drop_params or litellm.drop_params:
        verbose_logger.warning(
            "together_ai model %s does not support function calling per the model registry; dropping %s. Docs - %s",
            model,
            ", ".join(passed_tool_params),
            FUNCTION_CALLING_DOCS_URL,
        )
        return passed_tool_params
    raise UnsupportedParamsError(
        status_code=500,
        message=f"together_ai does not support parameters: {', '.join(passed_tool_params)}, for model={model}. To drop it from the call, set `litellm.drop_params = True`.",
    )


class TogetherAIChatConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> list:
        supports_fc: Final = _function_calling_verdict(model)
        supported_params: Final = super().get_supported_openai_params(model)
        if supports_fc is True:
            return supported_params
        verbose_logger.debug(
            "Only some together models support response_format. Docs - https://docs.together.ai/docs/function-calling"
        )
        return [  # mutable-ok: the inherited contract returns a plain list; building fresh avoids mutating the base class's value
            param for param in supported_params if param != "response_format"
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        mapped_openai_params: Final = super().map_openai_params(non_default_params, optional_params, model, drop_params)
        for param in _tool_params_to_drop(mapped_openai_params, model, drop_params):
            mapped_openai_params.pop(param)
        if mapped_openai_params.get("response_format") == PLAIN_TEXT_RESPONSE_FORMAT:
            mapped_openai_params.pop("response_format")
        return mapped_openai_params
