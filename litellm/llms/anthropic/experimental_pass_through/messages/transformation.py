from typing import Optional

from litellm.llms.base_llm.anthropic_messages.transformation import (
    BaseAnthropicMessagesConfig,
)

DEFAULT_ANTHROPIC_API_BASE = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_API_VERSION = "2023-06-01"


class AnthropicMessagesConfig(BaseAnthropicMessagesConfig):
    def get_supported_anthropic_messages_params(self, model: str) -> list:
        return [
            "messages",
            "model",
            "system",
            "max_tokens",
            "stop_sequences",
            "temperature",
            "top_p",
            "top_k",
            "tools",
            "tool_choice",
            "thinking",
            # TODO: Add Anthropic `metadata` support
            # "metadata",
        ]

    def get_complete_url(self, api_base: Optional[str], model: str) -> str:
        api_base = api_base or DEFAULT_ANTHROPIC_API_BASE
        if not api_base.endswith("/v1/messages"):
            api_base = f"{api_base}/v1/messages"
        return api_base

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        if "x-api-key" not in headers:
            headers["x-api-key"] = api_key
        if "anthropic-version" not in headers:
            headers["anthropic-version"] = DEFAULT_ANTHROPIC_API_VERSION
        if "content-type" not in headers:
            headers["content-type"] = "application/json"
        return headers
