"""
Translate from OpenAI's `/v1/chat/completions` to Foundry Local's `/v1/chat/completions`

Foundry Local (https://github.com/microsoft/Foundry-Local) runs LLMs on-device
and exposes an OpenAI-compatible REST API endpoint.
"""

from typing import Final

import litellm
from litellm.secret_managers.main import get_secret_str

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class FoundryLocalChatConfig(OpenAIGPTConfig):
    def get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        resolved_api_base: Final = api_base or get_secret_str("FOUNDRY_LOCAL_API_BASE") or litellm.api_base
        if not resolved_api_base:
            raise ValueError("Foundry Local requires api_base or FOUNDRY_LOCAL_API_BASE")
        dynamic_api_key: Final = api_key or get_secret_str("FOUNDRY_LOCAL_API_KEY") or "fake-api-key"
        return resolved_api_base, dynamic_api_key
