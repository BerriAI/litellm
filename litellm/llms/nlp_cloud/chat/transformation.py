import json
import time
from typing import TYPE_CHECKING, Any, List, Optional, Union

import httpx

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    convert_content_list_to_str,
)
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.types.llms.openai import AllMessageValues
from litellm.utils import ModelResponse, Usage

from ..common_utils import NLPCloudError

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    LoggingClass = LiteLLMLoggingObj
else:
    LoggingClass = Any


class NLPCloudConfig(BaseConfig):
    """
    Reference: https://docs.nlpcloud.com/#generation

    - `max_length` (int): Optional. The maximum number of tokens that the generated text should contain.

    - `length_no_input` (boolean): Optional. Whether `min_length` and `max_length` should not include the length of the input text.

    - `end_sequence` (string): Optional. A specific token that should be the end of the generated sequence.

    - `remove_end_sequence` (boolean): Optional. Whether to remove the `end_sequence` string from the result.

    - `remove_input` (boolean): Optional. Whether to remove the input text from the result.

    - `bad_words` (list of strings): Optional. List of tokens that are not allowed to be generated.

    - `temperature` (float): Optional. Temperature sampling. It modulates the next token probabilities.

    - `top_p` (float): Optional. Top P sampling. Below 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.

    - `top_k` (int): Optional. Top K sampling. The number of highest probability vocabulary tokens to keep for top k filtering.

    - `repetition_penalty` (float): Optional. Prevents the same word from being repeated too many times.

    - `num_beams` (int): Optional. Number of beams for beam search.

    - `num_return_sequences` (int): Optional. The number of independently computed returned sequences.
    """

    max_length: Optional[int] = None
    length_no_input: Optional[bool] = None
    end_sequence: Optional[str] = None
    remove_end_sequence: Optional[bool] = None
    remove_input: Optional[bool] = None
    bad_words: Optional[list] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    num_beams: Optional[int] = None
    num_return_sequences: Optional[int] = None

    def __init__(
        self,
        max_length: Optional[int] = None,
        length_no_input: Optional[bool] = None,
        end_sequence: Optional[str] = None,
        remove_end_sequence: Optional[bool] = None,
        remove_input: Optional[bool] = None,
        bad_words: Optional[list] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        num_beams: Optional[int] = None,
        num_return_sequences: Optional[int] = None,
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
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Token {api_key}"
        return headers

    def get_supported_openai_params(self, model: str) -> List:
        return [
            "max_tokens",
            "stream",
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "n",
            "stop",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "max_tokens":
                optional_params["max_length"] = value
            if param == "stream":
                optional_params["stream"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "top_p":
                optional_params["top_p"] = value
            if param == "presence_penalty":
                optional_params["presence_penalty"] = value
            if param == "frequency_penalty":
                optional_params["frequency_penalty"] = value
            if param == "n":
                optional_params["num_return_sequences"] = value
            if param == "stop":
                optional_params["stop_sequences"] = value
        return optional_params

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return NLPCloudError(
            status_code=status_code, message=error_message, headers=headers
        )

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        text = " ".join(convert_content_list_to_str(message) for message in messages)

        data = {
            "text": text,
            **optional_params,
        }

        return data

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LoggingClass,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        ## LOGGING
        logging_obj.post_call(
            input=None,
            api_key=api_key,
            original_response=raw_response.text,
            additional_args={"complete_input_dict": request_data},
        )

        ## RESPONSE OBJECT
        try:
            completion_response = raw_response.json()
        except Exception:
            raise NLPCloudError(
                message=raw_response.text, status_code=raw_response.status_code
            )
        if "error" in completion_response:
            raise NLPCloudError(
                message=completion_response["error"],
                status_code=raw_response.status_code,
            )
        else:
            try:
                if len(completion_response["generated_text"]) > 0:
                    model_response.choices[0].message.content = (  # type: ignore
                        completion_response["generated_text"]
                    )
            except Exception:
                raise NLPCloudError(
                    message=json.dumps(completion_response),
                    status_code=raw_response.status_code,
                )

        ## CALCULATING USAGE - baseten charges on time, not tokens - have some mapping of cost here.
        prompt_tokens = completion_response["nb_input_tokens"]
        completion_tokens = completion_response["nb_generated_tokens"]

        model_response.created = int(time.time())
        model_response.model = model
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        setattr(model_response, "usage", usage)
        return model_response
