import types
from typing import Callable, Literal, Optional, Union

import litellm


class VertexAILlama3Config:
    """
    Reference:https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/llama#streaming

    The class `VertexAILlama3Config` provides configuration for the VertexAI's Llama API interface. Below are the parameters:

    - `max_tokens` Required (integer) max tokens,

    Note: Please make sure to modify the default parameters as required for your use case.
    """

    max_tokens: Optional[int] = None

    def __init__(
        self,
        max_tokens: Optional[int] = None,
    ) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key == "max_tokens" and value is None:
                value = self.max_tokens
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
        return litellm.OpenAIConfig().get_supported_openai_params(model="gpt-3.5-turbo")

    def map_openai_params(
        self, non_default_params: dict, optional_params: dict, model: str
    ):
        if "max_completion_tokens" in non_default_params:
            non_default_params["max_tokens"] = non_default_params.pop(
                "max_completion_tokens"
            )
        return litellm.OpenAIConfig().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
        )
