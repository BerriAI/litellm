from litellm.llms.base_llm.chat.transformation import BaseConfig
import litellm
from typing import Optional, Tuple
from ...openai_like.chat.transformation import OpenAILikeChatConfig

from litellm.secret_managers.main import get_secret_str

class CloudRiftChatConfig(OpenAILikeChatConfig):
    @classmethod
    def get_config(cls):
        return super().get_config()

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "cloudrift"

    def _get_openai_compatible_provider_info(
            self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        # Cloudrift api is openai compatible
        api_base = (
                api_base
                or get_secret_str("CLOUDRIFT_API_BASE")
                or "https://inference.cloudrift.ai/v1"  # Default Lambda API base URL
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("CLOUDRIFT_API_KEY")
        return api_base, dynamic_api_key
