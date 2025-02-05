import re
import types
from enum import Enum
from typing import List, Optional, Union

import litellm
from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.llms.bedrock.chat.invoke_transformation import AmazonInvokeConfig


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
        locals_ = locals()
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


class AmazonAnthropicClaude3Config:
    """
    Reference:
        https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=claude
        https://docs.anthropic.com/claude/docs/models-overview#model-comparison

    Supported Params for the Amazon / Anthropic Claude 3 models:

    - `max_tokens` Required (integer) max tokens. Default is 4096
    - `anthropic_version` Required (string) version of anthropic for bedrock - e.g. "bedrock-2023-05-31"
    - `system` Optional (string) the system prompt, conversion from openai format to this is handled in factory.py
    - `temperature` Optional (float) The amount of randomness injected into the response
    - `top_p` Optional (float) Use nucleus sampling.
    - `top_k` Optional (int) Only sample from the top K options for each subsequent token
    - `stop_sequences` Optional (List[str]) Custom text sequences that cause the model to stop generating
    """

    max_tokens: Optional[int] = 4096  # Opus, Sonnet, and Haiku default
    anthropic_version: Optional[str] = "bedrock-2023-05-31"
    system: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        anthropic_version: Optional[str] = None,
    ) -> None:
        locals_ = locals()
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
            "max_tokens",
            "max_completion_tokens",
            "tools",
            "tool_choice",
            "stream",
            "stop",
            "temperature",
            "top_p",
            "extra_headers",
        ]

    def map_openai_params(self, non_default_params: dict, optional_params: dict):
        for param, value in non_default_params.items():
            if param == "max_tokens" or param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            if param == "tools":
                optional_params["tools"] = value
            if param == "stream":
                optional_params["stream"] = value
            if param == "stop":
                optional_params["stop_sequences"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "top_p":
                optional_params["top_p"] = value
        return optional_params


class AmazonAnthropicConfig:
    """
    Reference: https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=claude

    Supported Params for the Amazon / Anthropic models:

    - `max_tokens_to_sample` (integer) max tokens,
    - `temperature` (float) model temperature,
    - `top_k` (integer) top k,
    - `top_p` (integer) top p,
    - `stop_sequences` (string[]) list of stop sequences - e.g. ["\\n\\nHuman:"],
    - `anthropic_version` (string) version of anthropic for bedrock - e.g. "bedrock-2023-05-31"
    """

    max_tokens_to_sample: Optional[int] = litellm.max_tokens
    stop_sequences: Optional[list] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[int] = None
    anthropic_version: Optional[str] = None

    def __init__(
        self,
        max_tokens_to_sample: Optional[int] = None,
        stop_sequences: Optional[list] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[int] = None,
        anthropic_version: Optional[str] = None,
    ) -> None:
        locals_ = locals()
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

    def get_supported_openai_params(
        self,
    ):
        return [
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "stop",
            "top_p",
            "stream",
        ]

    def map_openai_params(self, non_default_params: dict, optional_params: dict):
        for param, value in non_default_params.items():
            if param == "max_tokens" or param == "max_completion_tokens":
                optional_params["max_tokens_to_sample"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "top_p":
                optional_params["top_p"] = value
            if param == "stop":
                optional_params["stop_sequences"] = value
            if param == "stream" and value is True:
                optional_params["stream"] = value
        return optional_params


class AmazonCohereConfig(AmazonInvokeConfig, BaseConfig):
    """
    Reference: https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=command

    Supported Params for the Amazon / Cohere models:

    - `max_tokens` (integer) max tokens,
    - `temperature` (float) model temperature,
    - `return_likelihood` (string) n/a
    """

    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    return_likelihood: Optional[str] = None

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        return_likelihood: Optional[str] = None,
    ) -> None:
        locals_ = locals()
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
        return [
            "max_tokens",
            "temperature",
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
            if k == "stream":
                optional_params["stream"] = v
            if k == "temperature":
                optional_params["temperature"] = v
            if k == "max_tokens":
                optional_params["max_tokens"] = v
        return optional_params


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
        locals_ = locals()
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


class AnthropicConstants(Enum):
    HUMAN_PROMPT = "\n\nHuman: "
    AI_PROMPT = "\n\nAssistant: "


class AmazonLlamaConfig(AmazonInvokeConfig, BaseConfig):
    """
    Reference: https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=meta.llama2-13b-chat-v1

    Supported Params for the Amazon / Meta Llama models:

    - `max_gen_len` (integer) max tokens,
    - `temperature` (float) temperature for model,
    - `top_p` (float) top p for model
    """

    max_gen_len: Optional[int] = None
    temperature: Optional[float] = None
    topP: Optional[float] = None

    def __init__(
        self,
        maxTokenCount: Optional[int] = None,
        temperature: Optional[float] = None,
        topP: Optional[int] = None,
    ) -> None:
        locals_ = locals()
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
                optional_params["max_gen_len"] = v
            if k == "temperature":
                optional_params["temperature"] = v
            if k == "top_p":
                optional_params["top_p"] = v
            if k == "stream":
                optional_params["stream"] = v
        return optional_params


class AmazonMistralConfig(AmazonInvokeConfig, BaseConfig):
    """
    Reference: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-mistral.html
    Supported Params for the Amazon / Mistral models:

    - `max_tokens` (integer) max tokens,
    - `temperature` (float) temperature for model,
    - `top_p` (float) top p for model
    - `stop` [string] A list of stop sequences that if generated by the model, stops the model from generating further output.
    - `top_k` (float) top k for model
    """

    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[float] = None
    stop: Optional[List[str]] = None

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[int] = None,
        top_k: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> None:
        locals_ = locals()
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
        return ["max_tokens", "temperature", "top_p", "stop", "stream"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for k, v in non_default_params.items():
            if k == "max_tokens":
                optional_params["max_tokens"] = v
            if k == "temperature":
                optional_params["temperature"] = v
            if k == "top_p":
                optional_params["top_p"] = v
            if k == "stop":
                optional_params["stop"] = v
            if k == "stream":
                optional_params["stream"] = v
        return optional_params
