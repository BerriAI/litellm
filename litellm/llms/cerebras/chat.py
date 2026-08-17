"""
Cerebras Chat Completions API

this is OpenAI compatible - no translation needed / occurs
"""

from typing import Final

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.utils import supports_reasoning


class CerebrasConfig(OpenAIGPTConfig):
    """
    Reference: https://inference-docs.cerebras.ai/api-reference/chat-completions

    Below are the parameters:
    """

    max_tokens: int | None = None
    response_format: dict | None = None
    seed: int | None = None
    stream: bool | None = None
    top_p: int | None = None
    tool_choice: str | None = None
    tools: list | None = None
    user: str | None = None
    reasoning_effort: str | None = None

    def __init__(
        self,
        max_tokens: int | None = None,
        response_format: dict | None = None,
        seed: int | None = None,
        stop: str | None = None,
        stream: bool | None = None,
        temperature: float | None = None,
        top_p: int | None = None,
        tool_choice: str | None = None,
        tools: list | None = None,
        user: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        locals_: Final = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the supported OpenAI params for the given model

        """

        supported_params: Final = [
            "max_tokens",
            "max_completion_tokens",
            "response_format",
            "seed",
            "stop",
            "stream",
            "temperature",
            "top_p",
            "tool_choice",
            "tools",
            "user",
        ]

        # Only add reasoning_effort for models that support it
        if supports_reasoning(model=model, custom_llm_provider="cerebras"):
            supported_params.append("reasoning_effort")

        return supported_params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_openai_params: Final = self.get_supported_openai_params(model=model)
        for param, value in non_default_params.items():
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_openai_params:
                optional_params[param] = value
        return optional_params
