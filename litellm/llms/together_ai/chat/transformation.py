"""
Translates from OpenAI's `/v1/chat/completions` to Together AI's `/v1/chat/completions`.

Docs: https://docs.together.ai/docs/chat-overview
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_logger
from litellm.utils import supports_function_calling, supports_reasoning

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

FUNCTION_CALLING_ONLY_PARAMS: Final = ("tools", "tool_choice", "function_call", "response_format")
PLAIN_TEXT_RESPONSE_FORMAT: Final = MappingProxyType({"type": "text"})

ADJUSTABLE_EFFORT_REASONING_MODELS: Final = frozenset(
    {
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    }
)
HYBRID_REASONING_MODELS: Final = frozenset(
    {
        "MiniMaxAI/MiniMax-M3",
        "Qwen/Qwen3.5-9B",
        "Qwen/Qwen3.6-Plus",
        "deepseek-ai/DeepSeek-V4-Pro",
        "moonshotai/Kimi-K3",
        "nvidia/nemotron-3-ultra-550b-a55b",
        "zai-org/GLM-5.2",
    }
)
HIGH_MAX_EFFORT_MODEL_PREFIX: Final = "deepseek-ai/DeepSeek-V4-Pro"
EFFORT_TRANSLATION: Final = MappingProxyType({"minimal": "low", "xhigh": "high", "max": "high"})
HIGH_MAX_EFFORT_TRANSLATION: Final = MappingProxyType(
    {"minimal": "high", "low": "high", "medium": "high", "high": "max", "xhigh": "max"}
)


class TogetherReasoningToggle(TypedDict):
    enabled: ReadOnly[bool]


def _supports_together_reasoning(model: str) -> bool:
    if model in ADJUSTABLE_EFFORT_REASONING_MODELS or model in HYBRID_REASONING_MODELS:
        return True
    if model.startswith(HIGH_MAX_EFFORT_MODEL_PREFIX):
        return True
    return supports_reasoning(model, custom_llm_provider="together_ai")


def _adjustable_effort(effort: str, model: str) -> str:
    if effort == "none":
        verbose_logger.debug(
            "together_ai model %s cannot disable reasoning; mapping reasoning_effort=none to low", model
        )
        return "low"
    return EFFORT_TRANSLATION.get(effort, effort)


def _reasoning_effort_payload(effort: str, model: str) -> Mapping[str, object]:
    if effort == "default":
        return MappingProxyType({})
    if model in ADJUSTABLE_EFFORT_REASONING_MODELS:
        return MappingProxyType({"reasoning_effort": _adjustable_effort(effort, model)})
    if effort == "none":
        disable_reasoning: Final[TogetherReasoningToggle] = {"enabled": False}
        return MappingProxyType({"reasoning": disable_reasoning})
    if model.startswith(HIGH_MAX_EFFORT_MODEL_PREFIX):
        return MappingProxyType({"reasoning_effort": HIGH_MAX_EFFORT_TRANSLATION.get(effort, effort)})
    return MappingProxyType({"reasoning_effort": EFFORT_TRANSLATION.get(effort, effort)})


class TogetherAIChatConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> list:
        supports_fc: bool | None = None
        try:
            supports_fc = supports_function_calling(model, custom_llm_provider="together_ai")
        except Exception as e:
            verbose_logger.debug("Error getting supported openai params: %s", e)

        supported_params: Final = super().get_supported_openai_params(model)
        if _supports_together_reasoning(model):
            supported_params.append("reasoning_effort")
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
        effort: Final = mapped_openai_params.get("reasoning_effort")
        if not isinstance(effort, str):
            return mapped_openai_params
        mapped_openai_params.pop("reasoning_effort")
        for key, value in _reasoning_effort_payload(effort, model).items():
            mapped_openai_params.setdefault(key, value)
        return mapped_openai_params
