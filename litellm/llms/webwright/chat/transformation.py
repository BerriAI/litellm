"""
Translate from OpenAI's `/v1/chat/completions` to WebWright's `/v1/chat/completions`
"""

from typing import Final

from litellm.secret_managers.main import get_secret_str

from ...openai_like.chat.transformation import OpenAILikeChatConfig


class WebwrightChatConfig(OpenAILikeChatConfig):
    """
    WebWright is OpenAI-compatible with standard endpoints.
    """

    @property
    def custom_llm_provider(self) -> str | None:
        return "webwright"

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        # WebWright is openai compatible, we just need to set the api_base
        resolved_api_base: Final = (
            api_base
            or get_secret_str("WEBWRIGHT_API_BASE")
            or "https://webwright.ai/v1"  # Default WebWright API base URL
        )
        dynamic_api_key: Final = api_key or get_secret_str("WEBWRIGHT_API_KEY")
        return resolved_api_base, dynamic_api_key

    def get_openai_compatible_provider_info(
        self,
        api_base: str | None,
        api_key: str | None,
    ) -> tuple[str | None, str | None]:
        return self._get_openai_compatible_provider_info(api_base, api_key)

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: matches the base provider signature that returns a plain list
        """
        WebWright supports the standard OpenAI chat completion parameters.
        Reference: https://webwright.ai/v1
        """
        return [
            "messages",  # Required
            "model",  # Required
            "stream",  # Optional
            "temperature",  # Optional
            "top_p",  # Optional
            "max_tokens",  # Optional
            "frequency_penalty",  # Optional
            "presence_penalty",  # Optional
            "stop",  # Optional
            "n",  # Optional
            "tools",  # Optional
            "tool_choice",  # Optional
            "response_format",  # Optional
            "seed",  # Optional
            "user",  # Optional
        ]
