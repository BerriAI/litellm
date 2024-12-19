"""
Sambanova Chat Completions API

this is OpenAI compatible - no translation needed / occurs
"""

from typing import Optional

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


class SambanovaConfig(OpenAIGPTConfig):
    """
    Reference: https://community.sambanova.ai/t/create-chat-completion-api/

    Below are the parameters:
    """

    max_tokens: Optional[int] = None
    response_format: Optional[dict] = None
    seed: Optional[int] = None
    stream: Optional[bool] = None
    top_p: Optional[int] = None
    tool_choice: Optional[str] = None
    tools: Optional[list] = None
    user: Optional[str] = None

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
        seed: Optional[int] = None,
        stop: Optional[str] = None,
        stream: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[int] = None,
        tool_choice: Optional[str] = None,
        tools: Optional[list] = None,
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
            "max_tokens",
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
