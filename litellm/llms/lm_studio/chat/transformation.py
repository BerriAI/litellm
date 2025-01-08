"""
Translate from OpenAI's `/v1/chat/completions` to LM Studio's `/chat/completions`
"""

from typing import Optional, Tuple

from litellm.secret_managers.main import get_secret_str

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class LMStudioChatConfig(OpenAIGPTConfig):
    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("LM_STUDIO_API_BASE")  # type: ignore
        dynamic_api_key = (
            api_key or get_secret_str("LM_STUDIO_API_KEY") or ""
        )  # vllm does not require an api key
        return api_base, dynamic_api_key
