import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, List, Optional, Union

import httpx

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, Message, ModelResponse, Usage

from ..common_utils import CohereError
from ..common_utils import ModelResponseIterator as CohereModelResponseIterator
from ..common_utils import validate_environment as cohere_validate_environment

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class CohereTextConfig(BaseConfig):
    """
    Reference: https://docs.cohere.com/reference/generate

    The class `CohereConfig` provides configuration for the Cohere's API interface. Below are the parameters:

    - `num_generations` (integer): Maximum number of generations returned. Default is 1, with a minimum value of 1 and a maximum value of 5.

    - `max_tokens` (integer): Maximum number of tokens the model will generate as part of the response. Default value is 20.

    - `truncate` (string): Specifies how the API handles inputs longer than maximum token length. Options include NONE, START, END. Default is END.

    - `temperature` (number): A non-negative float controlling the randomness in generation. Lower temperatures result in less random generations. Default is 0.75.

    - `preset` (string): Identifier of a custom preset, a combination of parameters such as prompt, temperature etc.

    - `end_sequences` (array of strings): The generated text gets cut at the beginning of the earliest occurrence of an end sequence, which will be excluded from the text.

    - `stop_sequences` (array of strings): The generated text gets cut at the end of the earliest occurrence of a stop sequence, which will be included in the text.

    - `k` (integer): Limits generation at each step to top `k` most likely tokens. Default is 0.

    - `p` (number): Limits generation at each step to most likely tokens with total probability mass of `p`. Default is 0.

    - `frequency_penalty` (number): Reduces repetitiveness of generated tokens. Higher values apply stronger penalties to previously occurred tokens.

    - `presence_penalty` (number): Reduces repetitiveness of generated tokens. Similar to frequency_penalty, but this penalty applies equally to all tokens that have already appeared.

    - `return_likelihoods` (string): Specifies how and if token likelihoods are returned with the response. Options include GENERATION, ALL and NONE.

    - `logit_bias` (object): Used to prevent the model from generating unwanted tokens or to incentivize it to include desired tokens. e.g. {"hello_world": 1233}
    """

    num_generations: Optional[int] = None
    max_tokens: Optional[int] = None
    truncate: Optional[str] = None
    temperature: Optional[int] = None
    preset: Optional[str] = None
    end_sequences: Optional[list] = None
    stop_sequences: Optional[list] = None
    k: Optional[int] = None
    p: Optional[int] = None
    frequency_penalty: Optional[int] = None
    presence_penalty: Optional[int] = None
    return_likelihoods: Optional[str] = None
    logit_bias: Optional[dict] = None

    def __init__(
        self,
        num_generations: Optional[int] = None,
        max_tokens: Optional[int] = None,
        truncate: Optional[str] = None,
        temperature: Optional[int] = None,
        preset: Optional[str] = None,
        end_sequences: Optional[list] = None,
        stop_sequences: Optional[list] = None,
        k: Optional[int] = None,
        p: Optional[int] = None,
        frequency_penalty: Optional[int] = None,
        presence_penalty: Optional[int] = None,
        return_likelihoods: Optional[str] = None,
        logit_bias: Optional[dict] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        return cohere_validate_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            api_key=api_key,
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return CohereError(status_code=status_code, message=error_message)

    def get_supported_openai_params(self, model: str) -> List:
        return [
            "stream",
            "temperature",
            "max_tokens",
            "logit_bias",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
            "n",
            "extra_headers",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "stream":
                optional_params["stream"] = value
            elif param == "temperature":
                optional_params["temperature"] = value
            elif param == "max_tokens":
                optional_params["max_tokens"] = value
            elif param == "n":
                optional_params["num_generations"] = value
            elif param == "logit_bias":
                optional_params["logit_bias"] = value
            elif param == "top_p":
                optional_params["p"] = value
            elif param == "frequency_penalty":
                optional_params["frequency_penalty"] = value
            elif param == "presence_penalty":
                optional_params["presence_penalty"] = value
            elif param == "stop":
                optional_params["stop_sequences"] = value
        return optional_params

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        prompt = " ".join(
            convert_content_list_to_str(message=message) for message in messages
        )

        ## Load Config
        config = litellm.CohereConfig.get_config()
        for k, v in config.items():
            if (
                k not in optional_params
            ):  # completion(top_k=3) > cohere_config(top_k=3) <- allows for dynamic variables to be passed in
                optional_params[k] = v

        ## Handle Tool Calling
        if "tools" in optional_params:
            _is_function_call = True
            tool_calling_system_prompt = self._construct_cohere_tool_for_completion_api(
                tools=optional_params["tools"]
            )
            optional_params["tools"] = tool_calling_system_prompt

        data = {
            "model": model,
            "prompt": prompt,
            **optional_params,
        }

        return data

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        prompt = " ".join(
            convert_content_list_to_str(message=message) for message in messages
        )
        completion_response = raw_response.json()
        choices_list = []
        for idx, item in enumerate(completion_response["generations"]):
            if len(item["text"]) > 0:
                message_obj = Message(content=item["text"])
            else:
                message_obj = Message(content=None)
            choice_obj = Choices(
                finish_reason=item["finish_reason"],
                index=idx + 1,
                message=message_obj,
            )
            choices_list.append(choice_obj)
        model_response.choices = choices_list  # type: ignore

        ## CALCULATING USAGE
        prompt_tokens = len(encoding.encode(prompt))
        completion_tokens = len(
            encoding.encode(model_response["choices"][0]["message"].get("content", ""))
        )

        model_response.created = int(time.time())
        model_response.model = model
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        setattr(model_response, "usage", usage)
        return model_response

    def _construct_cohere_tool_for_completion_api(
        self,
        tools: Optional[List] = None,
    ) -> dict:
        if tools is None:
            tools = []
        return {"tools": tools}

    def get_model_response_iterator(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str], ModelResponse],
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ):
        return CohereModelResponseIterator(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )
