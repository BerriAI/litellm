"""
Transform request from OpenAI format to Morph format.

[TODO] Docs: Morph supports the OpenAI API format.
https://docs.morphllm.com/quickstart
"""

from typing import Final

from litellm.secret_managers.main import get_secret_str

from ...openai_like.chat.transformation import OpenAILikeChatConfig

_MORPH_APPLY_MODELS = ("morph-v3-fast", "morph-v3-large")
_MORPH_COMMON_PARAMS = (
    "messages",
    "model",
    "stream",
    "temperature",
    "stop",
    "max_tokens",
)
_MORPH_CHAT_PARAMS = (
    "top_p",
    "frequency_penalty",
    "presence_penalty",
    "seed",
    "logit_bias",
    "tools",
    "response_format",
    "logprobs",
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
        api_base = (
            api_base or get_secret_str("MORPH_API_BASE") or "https://api.morphllm.com/v1"  # default api base
        )
        dynamic_api_key: Final = api_key or get_secret_str("MORPH_API_KEY")
        return api_base, dynamic_api_key

    def get_supported_openai_params(self, model: str) -> list[str]:
        model_params = () if model.split("/")[-1] in _MORPH_APPLY_MODELS else _MORPH_CHAT_PARAMS
        return list(_MORPH_COMMON_PARAMS + model_params)
