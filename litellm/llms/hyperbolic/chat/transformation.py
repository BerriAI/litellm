"""
Translate from OpenAI's `/v1/chat/completions` to Hyperbolic's `/v1/chat/completions`
"""

from typing import Optional, Tuple

from litellm.secret_managers.main import get_secret_str

from ...openai_like.chat.transformation import OpenAILikeChatConfig


class HyperbolicChatConfig(OpenAILikeChatConfig):
    """
    Hyperbolic is OpenAI-compatible with standard endpoints
    """

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "hyperbolic"

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        # Hyperbolic is openai compatible, we just need to set the api_base
        api_base = (
            api_base
            or get_secret_str("HYPERBOLIC_API_BASE")
            or "https://api.hyperbolic.xyz/v1"  # Default Hyperbolic API base URL
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("HYPERBOLIC_API_KEY")
        return api_base, dynamic_api_key

    def get_supported_openai_params(self, model: str) -> list:
        """
        Hyperbolic supports standard OpenAI parameters
        Reference: https://docs.hyperbolic.xyz/docs/rest-api
        """
        return [
            "messages",  # Required
            "model",  # Required
            "stream",  # Optional
            "temperature",  # Optional
            "top_p",  # Optional
            "max_tokens",  # Optional
            "frequency_penalty",  # Optional
            "presence_penalty",  # Optional
            "stop",  # Optional
            "n",  # Optional
            "tools",  # Optional
            "tool_choice",  # Optional
            "response_format",  # Optional
            "seed",  # Optional
            "user",  # Optional
        ]
