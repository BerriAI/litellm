"""
Translate from OpenAI's `/v1/chat/completions` to v0's `/v1/chat/completions`
"""

from typing import Optional, Tuple

from litellm.secret_managers.main import get_secret_str

from ...openai_like.chat.transformation import OpenAILikeChatConfig


class V0ChatConfig(OpenAILikeChatConfig):
    """
    v0 is OpenAI-compatible with standard endpoints
    """
    
    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "v0"

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        # v0 is openai compatible, we just need to set the api_base
        api_base = (
            api_base
            or get_secret_str("V0_API_BASE")
            or "https://api.v0.dev/v1"  # Default v0 API base URL
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("V0_API_KEY")
        return api_base, dynamic_api_key

    def get_supported_openai_params(self, model: str) -> list:
        """
        v0 supports a limited subset of OpenAI parameters
        Reference: https://v0.dev/docs/v0-model-api#request-body
        """
        return [
            "messages",     # Required
            "model",        # Required
            "stream",       # Optional
            "tools",        # Optional
            "tool_choice",  # Optional
        ]