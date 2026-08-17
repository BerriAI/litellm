"""
Translate from OpenAI's `/v1/audio/transcriptions` to Groq's `/v1/audio/transcriptions`
"""

import types
from typing import Final

import litellm


class GroqSTTConfig:
    frequency_penalty: int | None = None
    function_call: str | dict | None = None
    functions: list | None = None
    logit_bias: dict | None = None
    max_tokens: int | None = None
    n: int | None = None
    presence_penalty: int | None = None
    stop: str | list | None = None
    temperature: int | None = None
    top_p: int | None = None
    response_format: dict | None = None
    tools: list | None = None
    tool_choice: str | dict | None = None

    def __init__(
        self,
        frequency_penalty: int | None = None,
        function_call: str | dict | None = None,
        functions: list | None = None,
        logit_bias: dict | None = None,
        max_tokens: int | None = None,
        n: int | None = None,
        presence_penalty: int | None = None,
        stop: str | list | None = None,
        temperature: int | None = None,
        top_p: int | None = None,
        response_format: dict | None = None,
        tools: list | None = None,
        tool_choice: str | dict | None = None,
    ) -> None:
        locals_: Final = locals().copy()
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

    def get_supported_openai_response_formats_stt(self) -> list[str]:
        return ["json", "verbose_json", "text"]

    def map_openai_params_stt(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        response_formats: Final = self.get_supported_openai_response_formats_stt()
        for param, value in non_default_params.items():
            if param == "response_format":
                if value in response_formats:
                    optional_params[param] = value
                else:
                    if litellm.drop_params is True or drop_params is True:
                        pass
                    else:
                        raise litellm.utils.UnsupportedParamsError(
                            message=f"Groq doesn't support response_format={value}. To drop unsupported openai params from the call, set `litellm.drop_params = True`",
                            status_code=400,
                        )
            else:
                optional_params[param] = value
        return optional_params
