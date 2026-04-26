"""
FireworksAIMessagesConfig for Anthropic-compatible Messages API support.
https://docs.fireworks.ai/api-reference/anthropic-messages

"""

from typing import Any, List, Optional, Tuple

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.secret_managers.main import get_secret_str


class FireworksAIMessagesConfig(AnthropicMessagesConfig):
    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "fireworks_ai"

    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Tuple[dict, Optional[str]]:
        api_key = api_key or (
            get_secret_str("FIREWORKS_API_KEY")
            or get_secret_str("FIREWORKS_AI_API_KEY")
            or get_secret_str("FIREWORKSAI_API_KEY")
            or get_secret_str("FIREWORKS_AI_TOKEN")
        )
        if api_key and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {api_key}"
        if "content-type" not in headers:
            headers["content-type"] = "application/json"
        return headers, api_base

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        api_base = (
            api_base
            or get_secret_str("FIREWORKS_API_BASE")
            or "https://api.fireworks.ai/inference/v1"
        )
        api_base = api_base.rstrip("/")
        if not api_base.endswith("/v1/messages"):
            if api_base.endswith("/v1"):
                api_base = f"{api_base}/messages"
            else:
                api_base = f"{api_base}/v1/messages"
        return api_base
