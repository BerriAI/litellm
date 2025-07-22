import time
from typing import TYPE_CHECKING, Any, List, Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse, Usage

from ..common_utils import OobaboogaError

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

    LoggingClass = LiteLLMLoggingObj
else:
    LoggingClass = Any


class OobaboogaConfig(OpenAIGPTConfig):
    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Optional[Union[dict, httpx.Headers]] = None,
    ) -> BaseLLMException:
        return OobaboogaError(
            status_code=status_code, message=error_message, headers=headers
        )

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
            input=messages,
            api_key=api_key,
            original_response=raw_response.text,
            additional_args={"complete_input_dict": request_data},
        )

        ## RESPONSE OBJECT
        try:
            completion_response = raw_response.json()
        except Exception:
            raise OobaboogaError(
                message=raw_response.text, status_code=raw_response.status_code
            )
        if "error" in completion_response:
            raise OobaboogaError(
                message=completion_response["error"],
                status_code=raw_response.status_code,
            )
        else:
            try:
                model_response.choices[0].message.content = completion_response["choices"][0]["message"]["content"]  # type: ignore
            except Exception as e:
                raise OobaboogaError(
                    message=str(e),
                    status_code=raw_response.status_code,
                )

        model_response.created = int(time.time())
        model_response.model = model
        usage = Usage(
            prompt_tokens=completion_response["usage"]["prompt_tokens"],
            completion_tokens=completion_response["usage"]["completion_tokens"],
            total_tokens=completion_response["usage"]["total_tokens"],
        )
        setattr(model_response, "usage", usage)
        return model_response

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
        }
        if api_key is not None:
            headers["Authorization"] = f"Token {api_key}"
        return headers
