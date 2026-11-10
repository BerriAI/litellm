"""
MiniMax OpenAI transformation config - extends OpenAI chat config for MiniMax's OpenAI-compatible API
"""

from typing import Final

import litellm
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam


def _supports_reasoning(model: str) -> bool:
    try:
        return litellm.supports_reasoning(model=model, custom_llm_provider="minimax")
    except Exception:
        return False


class MinimaxChatConfig(OpenAIGPTConfig):
    """
    MiniMax OpenAI configuration that extends OpenAIGPTConfig.
    MiniMax provides an OpenAI-compatible API at:
    - International: https://api.minimax.io/v1
    - China: https://api.minimaxi.com/v1

    Supported models:
    - MiniMax-M2.1
    - MiniMax-M2.1-lightning
    - MiniMax-M2
    """

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        """
        Get MiniMax API key from environment or parameters.
        """
        return api_key or get_secret_str("MINIMAX_API_KEY") or litellm.api_key

    @staticmethod
    def get_api_base(
        api_base: str | None = None,
    ) -> str:
        """
        Get MiniMax API base URL.
        Defaults to international endpoint: https://api.minimax.io/v1
        For China, set to: https://api.minimaxi.com/v1
        """
        return api_base or get_secret_str("MINIMAX_API_BASE") or "https://api.minimax.io/v1"

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        """
        Get the complete URL for MiniMax OpenAI API.
        Override to ensure we use MiniMax's endpoint.
        """
        # Get the base URL (either provided or default MiniMax endpoint)
        base_url: Final = self.get_api_base(api_base=api_base).rstrip("/")

        # Ensure it ends with /chat/completions
        if base_url.endswith("/chat/completions"):
            return base_url
        if base_url.endswith("/v1"):
            return join_api_base(base_url, "chat/completions")
        return join_api_base(base_url, "v1/chat/completions")

    def remove_cache_control_flag_from_messages_and_tools(
        self,
        model: str,
        messages: list[AllMessageValues],
        tools: list[ChatCompletionToolParam] | None = None,
    ) -> tuple[list[AllMessageValues], list[ChatCompletionToolParam] | None]:
        """
        Override to preserve cache_control for MiniMax.
        MiniMax supports cache_control - don't strip it.
        """
        # MiniMax supports cache_control, so return messages and tools unchanged
        return messages, tools

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get supported OpenAI parameters for MiniMax.
        Adds reasoning_split, plus thinking and reasoning_effort when the cost map, or its
        minimax/ fallback rule for unlisted models, says the model supports reasoning.
        """
        reasoning_params: Final = ("thinking", "reasoning_effort") if _supports_reasoning(model) else ()
        return [*super().get_supported_openai_params(model=model), "reasoning_split", *reasoning_params]
