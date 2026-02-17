"""
Translates from OpenAI's `/v1/chat/completions` to Komilion's `/v1/chat/completions`

Komilion is an AI model router that automatically selects the best model for each
request. Instead of specifying a model name, users specify a routing tier:
- frugal: cheapest model that fits the task
- balanced: quality/cost sweet spot
- premium: frontier models only

Komilion implements the OpenAI-compatible API spec.

Docs: https://komilion.com/docs
"""

from typing import Optional, Tuple

from litellm.secret_managers.main import get_secret_str

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class KomilionChatConfig(OpenAIGPTConfig):

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = (
            api_base
            or get_secret_str("KOMILION_API_BASE")
            or "https://api.komilion.com/v1"
        )
        dynamic_api_key = api_key or get_secret_str("KOMILION_API_KEY")
        return api_base, dynamic_api_key

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "komilion"

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        if not api_base:
            api_base = "https://api.komilion.com/v1"

        if not api_base.endswith("/chat/completions"):
            api_base = f"{api_base}/chat/completions"

        return api_base
