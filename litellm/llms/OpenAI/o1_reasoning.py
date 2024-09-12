"""
Support for o1 model family 

https://platform.openai.com/docs/guides/reasoning

Translations handled by LiteLLM:
- modalities: image => drop param (if user opts in to dropping param)  
- role: system ==> translate to role 'user' 
- streaming => faked by LiteLLM 
- Tools, response_format =>  drop param (if user opts in to dropping param) 
- Logprobs => drop param (if user opts in to dropping param) 
"""

import types
from typing import Optional, Union

import litellm

from .openai import OpenAIConfig


class OpenAIO1Config(OpenAIConfig):
    """
    Reference: https://platform.openai.com/docs/guides/reasoning

    """

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

        all_openai_params = litellm.OpenAIConfig.get_supported_openai_params(
            model="gpt-4o"
        )
        non_supported_params = [
            "logprobs",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "function_call",
            "functions",
        ]

        return [
            param for param in all_openai_params if param not in non_supported_params
        ]

    def map_openai_params(self, non_default_params: dict, optional_params: dict):
        for param, value in non_default_params.items():
            if param == "max_tokens":
                optional_params["max_completion_tokens"] = value
        return optional_params
