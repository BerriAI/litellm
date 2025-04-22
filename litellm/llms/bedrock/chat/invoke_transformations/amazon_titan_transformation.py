import re
import types
from typing import List, Optional, Union

import litellm
from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
    AmazonInvokeConfig,
)


class AmazonTitanConfig(AmazonInvokeConfig, BaseConfig):
    """
    Reference: https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=titan-text-express-v1

    Supported Params for the Amazon Titan models:

    - `maxTokenCount` (integer) max tokens,
    - `stopSequences` (string[]) list of stop sequence strings
    - `temperature` (float) temperature for model,
    - `topP` (int) top p for model
    """

    maxTokenCount: Optional[int] = None
    stopSequences: Optional[list] = None
    temperature: Optional[float] = None
    topP: Optional[int] = None

    def __init__(
        self,
        maxTokenCount: Optional[int] = None,
        stopSequences: Optional[list] = None,
        temperature: Optional[float] = None,
        topP: Optional[int] = None,
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

    def _map_and_modify_arg(
        self,
        supported_params: dict,
        provider: str,
        model: str,
        stop: Union[List[str], str],
    ):
        """
        filter params to fit the required provider format, drop those that don't fit if user sets `litellm.drop_params = True`.
        """
        filtered_stop = None
        if "stop" in supported_params and litellm.drop_params:
            if provider == "bedrock" and "amazon" in model:
                filtered_stop = []
                if isinstance(stop, list):
                    for s in stop:
                        if re.match(r"^(\|+|User:)$", s):
                            filtered_stop.append(s)
        if filtered_stop is not None:
            supported_params["stop"] = filtered_stop

        return supported_params

    def get_supported_openai_params(self, model: str) -> List[str]:
        return [
            "max_tokens",
            "max_completion_tokens",
            "stop",
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
            if k == "max_tokens" or k == "max_completion_tokens":
                optional_params["maxTokenCount"] = v
            if k == "temperature":
                optional_params["temperature"] = v
            if k == "stop":
                filtered_stop = self._map_and_modify_arg(
                    {"stop": v}, provider="bedrock", model=model, stop=v
                )
                optional_params["stopSequences"] = filtered_stop["stop"]
            if k == "top_p":
                optional_params["topP"] = v
            if k == "stream":
                optional_params["stream"] = v
        return optional_params
