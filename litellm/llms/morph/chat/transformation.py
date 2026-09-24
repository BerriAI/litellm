"""
Transform request from OpenAI format to Morph format.

[TODO] Docs: Morph supports the OpenAI API format.
https://docs.morphllm.com/quickstart
"""

from typing import Final

import litellm
from litellm.secret_managers.main import get_secret_str

from ...openai_like.chat.transformation import OpenAILikeChatConfig

TOOL_CALLING_PARAMS: Final = frozenset({"tools", "tool_choice"})


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

    @staticmethod
    def _registry_disables_function_calling(model: str) -> bool:
        # Morph ships models faster than the registry is updated, so only an
        # explicit `supports_function_calling: false` entry withholds tools.
        registry_key: Final = model if model.startswith("morph/") else f"morph/{model}"
        model_info: Final = litellm.model_cost.get(registry_key)
        return isinstance(model_info, dict) and model_info.get("supports_function_calling") is False

    def get_supported_openai_params(self, model: str) -> list:
        supported_params: Final = [
            "extra_headers",
            "frequency_penalty",
            "logit_bias",
            "max_completion_tokens",
            "max_retries",
            "max_tokens",
            "messages",
            "model",
            "presence_penalty",
            "response_format",
            "seed",
            "stop",
            "stream",
            "stream_options",
            "temperature",
            "tool_choice",
            "tools",
            "top_p",
        ]
        if self._registry_disables_function_calling(model):
            return [param for param in supported_params if param not in TOOL_CALLING_PARAMS]
        return supported_params
