"""
AI21 Chat Completions API

this is OpenAI compatible - no translation needed / occurs
"""

from typing import Optional, Union

from ...openai_like.chat.transformation import OpenAILikeChatConfig


class AI21ChatConfig(OpenAILikeChatConfig):
    """
    Reference: https://docs.ai21.com/reference/jamba-15-api-ref#request-parameters

    Below are the parameters:
    """

    tools: Optional[list] = None
    response_format: Optional[dict] = None
    documents: Optional[list] = None
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, list]] = None
    n: Optional[int] = None
    stream: Optional[bool] = None
    seed: Optional[int] = None
    tool_choice: Optional[str] = None
    user: Optional[str] = None

    def __init__(
        self,
        tools: Optional[list] = None,
        response_format: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[Union[str, list]] = None,
        n: Optional[int] = None,
        stream: Optional[bool] = None,
        seed: Optional[int] = None,
        tool_choice: Optional[str] = None,
        user: Optional[str] = None,
    ) -> None:
        locals_ = locals().copy()
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
