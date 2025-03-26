from typing import Any, List, Optional, Union

from httpx import Headers, Response

import litellm
from litellm.llms.base_llm.chat.transformation import (
    BaseConfig,
    BaseLLMException,
    LiteLLMLoggingObj,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

from ..common_utils import PetalsError


class PetalsConfig(BaseConfig):
    """
    Reference: https://github.com/petals-infra/chat.petals.dev#post-apiv1generate
    The `PetalsConfig` class encapsulates the configuration for the Petals API. The properties of this class are described below:

    - `max_length` (integer): This represents the maximum length of the generated text (including the prefix) in tokens.

    - `max_new_tokens` (integer): This represents the maximum number of newly generated tokens (excluding the prefix).

    The generation parameters are compatible with `.generate()` from Hugging Face's Transformers library:

    - `do_sample` (boolean, optional): If set to 0 (default), the API runs greedy generation. If set to 1, the API performs sampling using the parameters below:

    - `temperature` (float, optional): This value sets the temperature for sampling.

    - `top_k` (integer, optional): This value sets the limit for top-k sampling.

    - `top_p` (float, optional): This value sets the limit for top-p (nucleus) sampling.

    - `repetition_penalty` (float, optional): This helps apply the repetition penalty during text generation, as discussed in this paper.
    """

    max_length: Optional[int] = None
    max_new_tokens: Optional[int] = (
        litellm.max_tokens
    )  # petals requires max tokens to be set
    do_sample: Optional[bool] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None

    def __init__(
        self,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[
            int
        ] = litellm.max_tokens,  # petals requires max tokens to be set
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, Headers]
    ) -> BaseLLMException:
        return PetalsError(
            status_code=status_code, message=error_message, headers=headers
        )

    def get_supported_openai_params(self, model: str) -> List:
        return ["max_tokens", "temperature", "top_p", "stream"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for param, value in non_default_params.items():
            if param == "max_tokens":
                optional_params["max_new_tokens"] = value
            if param == "temperature":
                optional_params["temperature"] = value
            if param == "top_p":
                optional_params["top_p"] = value
            if param == "stream":
                optional_params["stream"] = value
        return optional_params

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        raise NotImplementedError(
            "Petals transformation currently done in handler.py. [TODO] Move to the transformation.py"
        )

    def transform_response(
        self,
        model: str,
        raw_response: Response,
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
        raise NotImplementedError(
            "Petals transformation currently done in handler.py. [TODO] Move to the transformation.py"
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        return {}
