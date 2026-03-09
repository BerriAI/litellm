"""
Translate from OpenAI's `/v1/chat/completions` to Crusoe's `/v1/chat/completions`
"""

from typing import Optional, Tuple

from litellm.secret_managers.main import get_secret_str

from ...openai_like.chat.transformation import OpenAILikeChatConfig


class CrusoeChatConfig(OpenAILikeChatConfig):
    """
    Crusoe is OpenAI-compatible with standard endpoints.

    Docs: https://docs.crusoecloud.com/managed-inference/overview/index.html
    """

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "crusoe"

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = (
            api_base
            or get_secret_str("CRUSOE_API_BASE")
            or "https://managed-inference-api-proxy.crusoecloud.com/v1/"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("CRUSOE_API_KEY")
        return api_base, dynamic_api_key

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "messages",
            "model",
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
        ]
