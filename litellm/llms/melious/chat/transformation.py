"""
Translates from OpenAI's `/v1/chat/completions` to Melious' OpenAI-compatible endpoint.

Melious is a European AI gateway (https://melious.ai) fronting open-weight models.
"""

from typing import Final

from litellm.secret_managers.main import get_secret_str

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import MELIOUS_OPENAI_API_BASE, openai_api_base


class MeliousChatConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> str | None:
        return "melious"

    def _get_openai_compatible_provider_info(self, api_base: str | None, api_key: str | None) -> tuple[str, str | None]:
        resolved_api_base: Final = api_base or get_secret_str("MELIOUS_API_BASE") or MELIOUS_OPENAI_API_BASE
        resolved_api_key: Final = api_key or get_secret_str("MELIOUS_API_KEY")
        return openai_api_base(resolved_api_base), resolved_api_key
