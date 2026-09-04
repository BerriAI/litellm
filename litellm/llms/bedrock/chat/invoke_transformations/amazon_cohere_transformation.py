import types
from typing import Final

from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
    AmazonInvokeConfig,
)
from litellm.llms.cohere.chat.transformation import CohereChatConfig


class AmazonCohereConfig(AmazonInvokeConfig, CohereChatConfig):
    """
    Reference: https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=command

    Supported Params for the Amazon / Cohere models:

    - `max_tokens` (integer) max tokens,
    - `temperature` (float) model temperature,
    - `return_likelihood` (string) n/a
    """

    max_tokens: int | None = None
    return_likelihood: str | None = None

    def __init__(
        self,
        max_tokens: int | None = None,
        temperature: float | None = None,
        return_likelihood: str | None = None,
    ) -> None:
        locals_: Final = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

        AmazonInvokeConfig.__init__(self)

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not k.startswith("_abc")
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

    def get_supported_openai_params(self, model: str) -> list[str]:
        supported_params: Final = CohereChatConfig.get_supported_openai_params(self, model=model)
        return supported_params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        return CohereChatConfig.map_openai_params(
            self,
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )
