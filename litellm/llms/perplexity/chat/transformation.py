"""
Translate from OpenAI's `/v1/chat/completions` to Perplexity's `/v1/chat/completions`
"""

from typing import Optional, Tuple

from litellm.secret_managers.main import get_secret_str

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class PerplexityChatConfig(OpenAIGPTConfig):
    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("PERPLEXITY_API_BASE") or "https://api.perplexity.ai"  # type: ignore
        dynamic_api_key = (
            api_key
            or get_secret_str("PERPLEXITYAI_API_KEY")
            or get_secret_str("PERPLEXITY_API_KEY")
        )
        return api_base, dynamic_api_key

    def get_supported_openai_params(self, model: str) -> list:
        """
        Perplexity supports a subset of OpenAI params

        Ref: https://docs.perplexity.ai/api-reference/chat-completions

        Eg. Perplexity does not support tools, tool_choice, function_call, functions, etc.
        """
        return [
            "frequency_penalty",
            "max_tokens",
            "max_completion_tokens",
            "presence_penalty",
            "response_format",
            "stream",
            "temperature",
            "top_p",
            "max_retries",
            "extra_headers",
        ]
