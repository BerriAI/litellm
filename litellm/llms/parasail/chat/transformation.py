"""
Parasail chat completion transformation
"""
from typing import Optional, Tuple

from litellm.secret_managers.main import get_secret_str

from ...openai_like.chat.transformation import OpenAILikeChatConfig


class ParasailChatConfig(OpenAILikeChatConfig):
    """
    Configuration for Parasail chat completions.
    Parasail is OpenAI-compatible, so we extend OpenAILikeChatConfig.
    """

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the API base URL and API key for Parasail.
        """
        api_base = (
            api_base
            or get_secret_str("PARASAIL_API_BASE")
            or "https://api.parasail.io/v1"
        )
        dynamic_api_key = api_key or get_secret_str("PARASAIL_API_KEY")
        return api_base, dynamic_api_key

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the list of OpenAI parameters supported by Parasail.
        Since Parasail is OpenAI-compatible, we support standard OpenAI parameters.
        """
        return [
            "frequency_penalty",
            "function_call",
            "functions", 
            "logit_bias",
            "max_tokens",
            "n",
            "presence_penalty",
            "stop",
            "stream",
            "temperature",
            "tool_choice",
            "tools",
            "top_p",
            "response_format",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool = False,
        replace_max_completion_tokens_with_max_tokens: bool = True,
    ) -> dict:
        """
        Map OpenAI parameters to Parasail format.
        Since Parasail is OpenAI-compatible, minimal transformation needed.
        """
        return super().map_openai_params(
            non_default_params, 
            optional_params, 
            model, 
            drop_params,
            replace_max_completion_tokens_with_max_tokens
        )
