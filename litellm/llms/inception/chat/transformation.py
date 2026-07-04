"""
Translate from OpenAI's `/v1/chat/completions` to Inception's `/v1/chat/completions`

Inception Labs (https://www.inceptionlabs.ai) serves the Mercury family of
diffusion LLMs through an OpenAI-compatible API, so we only need to point the
OpenAI-like handler at the Inception API base and pick up the Inception API key.
"""

from typing import List, Optional, Tuple

import litellm
from litellm.secret_managers.main import get_secret_str

from ...openai_like.chat.transformation import OpenAILikeChatConfig


class InceptionChatConfig(OpenAILikeChatConfig):
    """
    Inception is OpenAI-compatible with standard endpoints
    """

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "inception"

    def get_supported_openai_params(self, model: str) -> List:
        return [
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "stop",
            "tools",
            "tool_choice",
            "stream",
            "stream_options",
            "response_format",
            "reasoning_effort",
            "reasoning_summary",
            "reasoning_summary_wait",
            "diffusing",
            "realtime",
        ]

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        passed_api_base = api_base
        api_base = api_base or get_secret_str("INCEPTION_API_BASE") or "https://api.inceptionlabs.ai/v1"  # type: ignore
        dynamic_api_key = api_key
        if passed_api_base is None or api_key:
            dynamic_api_key = api_key or litellm.inception_key or get_secret_str("INCEPTION_API_KEY")
        return api_base, dynamic_api_key
