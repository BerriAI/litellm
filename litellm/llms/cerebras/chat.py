"""
Cerebras Chat Completions API

this is OpenAI compatible - no translation needed / occurs
"""

import types
from typing import Optional, Union


class CerebrasConfig:
    """
    Reference: https://inference-docs.cerebras.ai/api-reference/chat-completions

    Below are the parameters:
    """

    max_tokens: Optional[int] = None
    response_format: Optional[dict] = None
    seed: Optional[int] = None
    stop: Optional[str] = None
    stream: Optional[bool] = None
    temperature: Optional[float] = None
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
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the supported OpenAI params for the given model

        """

        return [
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

    def map_openai_params(
        self, model: str, non_default_params: dict, optional_params: dict
    ) -> dict:
        supported_openai_params = self.get_supported_openai_params(model=model)
        for param, value in non_default_params.items():
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_openai_params:
                optional_params[param] = value
        return optional_params
