from typing import Optional, Tuple

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str


class ParasailConfig(OpenAIGPTConfig):
    """
    Reference: Parasail is OpenAI compatible.

    API Key: PARASAIL_API_KEY
    Default API Base: https://api.parasail.io/v1
    """

    API_BASE_URL = "https://api.parasail.io/v1"

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "parasail"

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return api_key or get_secret_str("PARASAIL_API_KEY")

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> Optional[str]:
        return (
            api_base
            or get_secret_str("PARASAIL_API_BASE")
            or ParasailConfig.API_BASE_URL
        )

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        resolved_api_base = ParasailConfig.get_api_base(api_base)
        resolved_api_key = ParasailConfig.get_api_key(api_key)
        return resolved_api_base, resolved_api_key

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "max_tokens",
            "max_completion_tokens",
            "n",
            "temperature",
            "top_p",
            "seed",
            "stream",
            "stream_options",
            "logprobs",
            "top_logprobs",
            "frequency_penalty",
            "presence_penalty",
            "response_format",
            "stop",
            "logit_bias",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "user",
        ]
