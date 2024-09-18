"""
Transformation logic from OpenAI /v1/chat/completion format to Mistral's /chat/completion format.

Why separate file? Make it easy to see how transformation works

Docs - https://docs.mistral.ai/api/
"""

import types
from typing import List, Literal, Optional, Union


class MistralConfig:
    """
    Reference: https://docs.mistral.ai/api/

    The class `MistralConfig` provides configuration for the Mistral's Chat API interface. Below are the parameters:

    - `temperature` (number or null): Defines the sampling temperature to use, varying between 0 and 2. API Default - 0.7.

    - `top_p` (number or null): An alternative to sampling with temperature, used for nucleus sampling. API Default - 1.

    - `max_tokens` (integer or null): This optional parameter helps to set the maximum number of tokens to generate in the chat completion. API Default - null.

    - `tools` (list or null): A list of available tools for the model. Use this to specify functions for which the model can generate JSON inputs.

    - `tool_choice` (string - 'auto'/'any'/'none' or null): Specifies if/how functions are called. If set to none the model won't call a function and will generate a message instead. If set to auto the model can choose to either generate a message or call a function. If set to any the model is forced to call a function. Default - 'auto'.

    - `stop` (string or array of strings): Stop generation if this token is detected. Or if one of these tokens is detected when providing an array

    - `random_seed` (integer or null): The seed to use for random sampling. If set, different calls will generate deterministic results.

    - `safe_prompt` (boolean): Whether to inject a safety prompt before all conversations. API Default - 'false'.

    - `response_format` (object or null): An object specifying the format that the model must output. Setting to { "type": "json_object" } enables JSON mode, which guarantees the message the model generates is in JSON. When using JSON mode you MUST also instruct the model to produce JSON yourself with a system or a user message.
    """

    temperature: Optional[int] = None
    top_p: Optional[int] = None
    max_tokens: Optional[int] = None
    tools: Optional[list] = None
    tool_choice: Optional[Literal["auto", "any", "none"]] = None
    random_seed: Optional[int] = None
    safe_prompt: Optional[bool] = None
    response_format: Optional[dict] = None
    stop: Optional[Union[str, list]] = None

    def __init__(
        self,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[list] = None,
        tool_choice: Optional[Literal["auto", "any", "none"]] = None,
        random_seed: Optional[int] = None,
        safe_prompt: Optional[bool] = None,
        response_format: Optional[dict] = None,
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
            "max_tokens",
            "tools",
            "tool_choice",
            "seed",
            "stop",
            "response_format",
        ]

    def _map_tool_choice(self, tool_choice: str) -> str:
        if tool_choice == "auto" or tool_choice == "none":
            return tool_choice
        elif tool_choice == "required":
            return "any"
        else:  # openai 'tool_choice' object param not supported by Mistral API
            return "any"

    def map_openai_params(self, non_default_params: dict, optional_params: dict):
        for param, value in non_default_params.items():
            if param == "max_tokens":
                optional_params["max_tokens"] = value
            if param == "tools":
                optional_params["tools"] = value
            if param == "stream" and value is True:
                optional_params["stream"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "top_p":
                optional_params["top_p"] = value
            if param == "stop":
                optional_params["stop"] = value
            if param == "tool_choice" and isinstance(value, str):
                optional_params["tool_choice"] = self._map_tool_choice(
                    tool_choice=value
                )
            if param == "seed":
                optional_params["extra_body"] = {"random_seed": value}
            if param == "response_format":
                optional_params["response_format"] = value
        return optional_params
