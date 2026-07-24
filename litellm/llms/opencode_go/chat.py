"""
OpenCode Go Chat Completions API

OpenAI-compatible by default. Some models route through an Anthropic-style
endpoint (indicated by provider.npm == "@ai-sdk/anthropic" in the models.dev
catalog) — that routing is NOT handled in this file; see issue #31568.
"""

from typing import Optional

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


class OpenCodeGoConfig(OpenAIGPTConfig):
    """
    Reference: https://opencode.ai/docs
    """

    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: Optional[bool] = None
    tools: Optional[list] = None
    tool_choice: Optional[str] = None

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "max_tokens",
            "temperature",
            "stream",
            "tools",
            "tool_choice",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_openai_params = self.get_supported_openai_params(model=model)
        for param, value in non_default_params.items():
            if param in supported_openai_params:
                optional_params[param] = value
        return optional_params
