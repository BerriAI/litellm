import types
from typing import List, Optional

from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
    AmazonInvokeConfig,
)


class AmazonAI21Config(AmazonInvokeConfig, BaseConfig):
    """
    Reference: https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=j2-ultra

    Supported Params for the Amazon / AI21 models:

    - `maxTokens` (int32): The maximum number of tokens to generate per result. Optional, default is 16. If no `stopSequences` are given, generation stops after producing `maxTokens`.

    - `temperature` (float): Modifies the distribution from which tokens are sampled. Optional, default is 0.7. A value of 0 essentially disables sampling and results in greedy decoding.

    - `topP` (float): Used for sampling tokens from the corresponding top percentile of probability mass. Optional, default is 1. For instance, a value of 0.9 considers only tokens comprising the top 90% probability mass.

    - `stopSequences` (array of strings): Stops decoding if any of the input strings is generated. Optional.

    - `frequencyPenalty` (object): Placeholder for frequency penalty object.

    - `presencePenalty` (object): Placeholder for presence penalty object.

    - `countPenalty` (object): Placeholder for count penalty object.
    """

    maxTokens: Optional[int] = None
    temperature: Optional[float] = None
    topP: Optional[float] = None
    stopSequences: Optional[list] = None
    frequencePenalty: Optional[dict] = None
    presencePenalty: Optional[dict] = None
    countPenalty: Optional[dict] = None

    def __init__(
        self,
        maxTokens: Optional[int] = None,
        temperature: Optional[float] = None,
        topP: Optional[float] = None,
        stopSequences: Optional[list] = None,
        frequencePenalty: Optional[dict] = None,
        presencePenalty: Optional[dict] = None,
        countPenalty: Optional[dict] = None,
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

    def get_supported_openai_params(self, model: str) -> List:
        return [
            "max_tokens",
            "temperature",
            "top_p",
            "stream",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for k, v in non_default_params.items():
            if k == "max_tokens":
                optional_params["maxTokens"] = v
            if k == "temperature":
                optional_params["temperature"] = v
            if k == "top_p":
                optional_params["topP"] = v
            if k == "stream":
                optional_params["stream"] = v
        return optional_params
