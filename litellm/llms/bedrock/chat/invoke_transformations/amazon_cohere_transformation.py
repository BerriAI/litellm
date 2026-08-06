import types
from typing import List, Optional

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

    max_tokens: Optional[int] = None
    return_likelihood: Optional[str] = None

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_likelihood: Optional[str] = None,
    ) -> None:
        locals_ = locals().copy()
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

    def get_supported_openai_params(self, model: str) -> List[str]:
        supported_params = CohereChatConfig.get_supported_openai_params(self, model=model)
        return supported_params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        # Bedrock's Cohere Command models use Cohere's Generate API (not the
        # Chat API CohereChatConfig otherwise targets), which genuinely
        # supports `num_generations` -- see
        # https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-cohere-command.html
        # So `n` is excluded here before delegating, and mapped directly
        # instead of going through CohereChatConfig's Chat-API-specific
        # `n` handling (which would incorrectly raise/drop it for Bedrock).
        non_default_params_without_n = {k: v for k, v in non_default_params.items() if k != "n"}
        optional_params = CohereChatConfig.map_openai_params(
            self,
            non_default_params=non_default_params_without_n,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )
        if "n" in non_default_params:
            optional_params["num_generations"] = non_default_params["n"]
        return optional_params
