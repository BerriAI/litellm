"""
Translate from OpenAI's `/v1/chat/completions` to Malachi's compatible endpoint.
"""

from typing import Optional, Tuple

from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import LlmProviders

from ...openai_like.chat.transformation import OpenAILikeChatConfig


class MalachiChatConfig(OpenAILikeChatConfig):
    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.MALACHI

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("MALACHI_API_BASE")
        dynamic_api_key = api_key or get_secret_str("MALACHI_API_KEY")
        return api_base, dynamic_api_key
