import types
from typing import Literal, Optional, Union

import litellm


class VolcEngineConfig:
    frequency_penalty: Optional[int] = None
    function_call: Optional[Union[str, dict]] = None
    functions: Optional[list] = None
    logit_bias: Optional[dict] = None
    max_tokens: Optional[int] = None
    n: Optional[int] = None
    presence_penalty: Optional[int] = None
    stop: Optional[Union[str, list]] = None
    temperature: Optional[int] = None
    top_p: Optional[int] = None
    response_format: Optional[dict] = None

    def __init__(
        self,
        frequency_penalty: Optional[int] = None,
        function_call: Optional[Union[str, dict]] = None,
        functions: Optional[list] = None,
        logit_bias: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        n: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        stop: Optional[Union[str, list]] = None,
        temperature: Optional[int] = None,
        top_p: Optional[int] = None,
        response_format: Optional[dict] = None,
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
        return [
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "max_completion_tokens",
            "max_tokens",
            "n",
            "presence_penalty",
            "seed",
            "stop",
            "stream",
            "stream_options",
            "temperature",
            "top_p",
            "tools",
            "tool_choice",
            "function_call",
            "functions",
            "max_retries",
            "extra_headers",
        ]  # works across all models

    def map_openai_params(
        self, non_default_params: dict, optional_params: dict, model: str
    ) -> dict:
        supported_openai_params = self.get_supported_openai_params(model)
        for param, value in non_default_params.items():
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_openai_params:
                optional_params[param] = value
        return optional_params
