"""
Translate from OpenAI's `/v1/chat/completions` to AIHubMix's `/v1/chat/completions`

AIHubMix is an OpenAI-compatible aggregation API gateway.
Base URL: https://aihubmix.com/v1
"""

from typing import Optional, Tuple

from litellm.secret_managers.main import get_secret_str

from ...openai_like.chat.transformation import OpenAILikeChatConfig


class AIHubMixChatConfig(OpenAILikeChatConfig):

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "aihubmix"

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = (
            api_base
            or get_secret_str("AIHUBMIX_API_BASE")
            or "https://aihubmix.com/v1"
        )
        dynamic_api_key = api_key or get_secret_str("AIHUBMIX_API_KEY")
        return api_base, dynamic_api_key
