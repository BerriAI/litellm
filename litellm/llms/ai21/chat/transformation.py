"""
AI21 Chat Completions API

this is OpenAI compatible - no translation needed / occurs
"""

from typing import Final

from ...openai_like.chat.transformation import OpenAILikeChatConfig


class AI21ChatConfig(OpenAILikeChatConfig):
    """
    Reference: https://docs.ai21.com/reference/jamba-15-api-ref#request-parameters

    Below are the parameters:
    """

    tools: list | None = None
    response_format: dict | None = None
    documents: list | None = None
    max_tokens: int | None = None
    stop: str | list | None = None
    n: int | None = None
    stream: bool | None = None
    seed: int | None = None
    tool_choice: str | None = None
    user: str | None = None

    def __init__(
        self,
        tools: list | None = None,
        response_format: dict | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: str | list | None = None,
        n: int | None = None,
        stream: bool | None = None,
        seed: int | None = None,
        tool_choice: str | None = None,
        user: str | None = None,
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

        return [
            "tools",
            "response_format",
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "stop",
            "n",
            "stream",
            "seed",
            "tool_choice",
        ]
