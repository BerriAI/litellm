"""
Translate from OpenAI's `/v1/chat/completions` to Foundry Local's `/v1/chat/completions`

Foundry Local (https://github.com/microsoft/Foundry-Local) runs LLMs on-device
and exposes an OpenAI-compatible REST API endpoint.
"""

from typing import Optional, Tuple

from litellm.secret_managers.main import get_secret_str

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class FoundryLocalChatConfig(OpenAIGPTConfig):
    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("FOUNDRY_LOCAL_API_BASE")  # type: ignore
        dynamic_api_key = (
            api_key or get_secret_str("FOUNDRY_LOCAL_API_KEY") or "fake-api-key"
        )  # Foundry Local does not require an api key
        return api_base, dynamic_api_key
