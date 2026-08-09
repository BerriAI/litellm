"""
Transform request from OpenAI format to Morph format.

[TODO] Docs: Morph supports the OpenAI API format.
https://docs.morphllm.com/quickstart
"""

from typing import Final

from litellm.secret_managers.main import get_secret_str

from ...openai_like.chat.transformation import OpenAILikeChatConfig

_MORPH_APPLY_MODELS: Final = ("morph-v3-fast", "morph-v3-large")
_MORPH_LOGPROBS_MODELS: Final = (
    "morph-qwen35-397b",
    "morph-minimax3-428b",
    "morph-qwen36-27b",
    "morph-gemma4-31b",
    "morph-kimik3",
    "morph-kimik3-fast",
)
_MORPH_SERVICE_TIER_MODELS: Final = ("morph-glm52-744b",)
_MORPH_COMMON_PARAMS: Final = (
    "messages",
    "model",
    "stream",
    "temperature",
    "stop",
    "max_tokens",
)
_MORPH_CHAT_PARAMS: Final = (
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "logit_bias",
    "tools",
    "response_format",
)


class MorphChatConfig(OpenAILikeChatConfig):
    """
    Transform request from OpenAI format to Morph format.
    """

    @property
    def custom_llm_provider(self) -> str | None:
        return "morph"

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        resolved_api_base: Final = (
            api_base or get_secret_str("MORPH_API_BASE") or "https://api.morphllm.com/v1"  # default api base
        )
        dynamic_api_key: Final = api_key or get_secret_str("MORPH_API_KEY")
        return resolved_api_base, dynamic_api_key

    def get_supported_openai_params(self, model: str) -> list[str]:
        model_id: Final = model.split("/")[-1]
        model_params: Final = () if model_id in _MORPH_APPLY_MODELS else _MORPH_CHAT_PARAMS
        conditional_params: Final = (("logprobs",) if model_id in _MORPH_LOGPROBS_MODELS else ()) + (
            ("service_tier",) if model_id in _MORPH_SERVICE_TIER_MODELS else ()
        )
        return [*_MORPH_COMMON_PARAMS, *model_params, *conditional_params]
