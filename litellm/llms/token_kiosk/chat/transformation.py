"""
Translates from OpenAI's /v1/chat/completions to Token Kiosk's /v1/chat/completions
"""

from typing import Final

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str


class TokenKioskConfig(OpenAIGPTConfig):
    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        final_api_base: Final = api_base or get_secret_str("TOKEN_KIOSK_API_BASE") or "https://agent-router.gaib.ai/v1"
        dynamic_api_key: Final = api_key or get_secret_str("TOKEN_KIOSK_API_KEY")
        return final_api_base, dynamic_api_key

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,  # mutable-ok: OpenAIGPTConfig.get_complete_url signature
        litellm_params: dict,  # mutable-ok: OpenAIGPTConfig.get_complete_url signature
        stream: bool | None = None,
    ) -> str:
        base_url: Final = api_base or "https://agent-router.gaib.ai/v1"
        if base_url.endswith("/chat/completions"):
            return base_url
        if base_url.endswith("/"):
            return f"{base_url}chat/completions"
        return f"{base_url}/chat/completions"
