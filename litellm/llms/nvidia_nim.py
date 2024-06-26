"""
Nvidia NIM endpoint: https://docs.api.nvidia.com/nim/reference/databricks-dbrx-instruct-infer 

This is OpenAI compatible 

This file only contains param mapping logic

API calling is done using the OpenAI SDK with an api_base
"""

import types
from typing import Optional, Union


class NvidiaNimConfig:
    """
    Reference: https://docs.api.nvidia.com/nim/reference/databricks-dbrx-instruct-infer

    The class `NvidiaNimConfig` provides configuration for the Nvidia NIM's Chat Completions API interface. Below are the parameters:
    """

    temperature: Optional[int] = None
    top_p: Optional[int] = None
    frequency_penalty: Optional[int] = None
    presence_penalty: Optional[int] = None
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, list]] = None

    def __init__(
        self,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        frequency_penalty: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, list]] = None,
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

    def get_supported_openai_params(self):
        return [
            "stream",
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "max_tokens",
            "stop",
        ]

    def map_openai_params(
        self, non_default_params: dict, optional_params: dict
    ) -> dict:
        supported_openai_params = self.get_supported_openai_params()
        for param, value in non_default_params.items():
            if param in supported_openai_params:
                optional_params[param] = value
        return optional_params
