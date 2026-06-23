from typing import Optional, Tuple

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str


class AtlasCloudConfig(OpenAIGPTConfig):
    """
    Reference: Atlas Cloud is OpenAI compatible.
    Docs: https://www.atlascloud.ai/docs
    API Key: ATLASCLOUD_API_KEY
    Default API Base: https://api.atlascloud.ai/v1
    """

    API_BASE_URL = "https://api.atlascloud.ai/v1"

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "atlas_cloud"

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return api_key or get_secret_str("ATLASCLOUD_API_KEY")

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> Optional[str]:
        return (
            api_base
            or get_secret_str("ATLASCLOUD_API_BASE")
            or AtlasCloudConfig.API_BASE_URL
        )

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        # This method is called by get_llm_provider to resolve api_base and api_key
        resolved_api_base = AtlasCloudConfig.get_api_base(api_base)
        resolved_api_key = AtlasCloudConfig.get_api_key(api_key)
        return resolved_api_base, resolved_api_key

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "max_tokens",
            "n",
            "temperature",
            "top_p",
            "stream",
            "logprobs",
            "top_logprobs",
            "frequency_penalty",
            "presence_penalty",
            "response_format",
            "stop",
            "logit_bias",
            "tools",
            "tool_choice",
            "seed",
        ]
