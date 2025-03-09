"""
Translate from OpenAI's `/v1/audio/transcriptions` to Groq's `/v1/audio/transcriptions`
"""

import types
from typing import List, Optional, Union

import litellm


class GroqSTTConfig:

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
    tools: Optional[list] = None
    tool_choice: Optional[Union[str, dict]] = None

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
        tools: Optional[list] = None,
        tool_choice: Optional[Union[str, dict]] = None,
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

    def get_supported_openai_params_stt(self):
        return [
            "prompt",
            "response_format",
            "temperature",
            "language",
        ]

    def get_supported_openai_response_formats_stt(self) -> List[str]:
        return ["json", "verbose_json", "text"]

    def map_openai_params_stt(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        response_formats = self.get_supported_openai_response_formats_stt()
        for param, value in non_default_params.items():
            if param == "response_format":
                if value in response_formats:
                    optional_params[param] = value
                else:
                    if litellm.drop_params is True or drop_params is True:
                        pass
                    else:
                        raise litellm.utils.UnsupportedParamsError(
                            message="Groq doesn't support response_format={}. To drop unsupported openai params from the call, set `litellm.drop_params = True`".format(
                                value
                            ),
                            status_code=400,
                        )
            else:
                optional_params[param] = value
        return optional_params
