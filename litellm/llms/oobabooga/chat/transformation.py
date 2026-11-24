import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse, Usage

from ..common_utils import OobaboogaError

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer

    LoggingClass = LiteLLMLoggingObj
else:
    LoggingClass = Any


class OobaboogaConfig(OpenAIGPTConfig):
    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | httpx.Headers | None = None,
    ) -> BaseLLMException:
        return OobaboogaError(status_code=status_code, message=error_message, headers=headers)

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LoggingClass,
        request_data: Mapping[str, object],
        messages: list[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
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
            completion_response: Final = raw_response.json()
        except Exception:
            raise OobaboogaError(message=raw_response.text, status_code=raw_response.status_code)
        if "error" in completion_response:
            raise OobaboogaError(
                message=completion_response["error"],
                status_code=raw_response.status_code,
            )
        else:
            try:
                model_response.choices[0].message.content = completion_response["choices"][0]["message"]["content"]
            except Exception as e:
                raise OobaboogaError(
                    message=str(e),
                    status_code=raw_response.status_code,
                )

        model_response.created = int(time.time())
        model_response.model = model
        validated_counts: Final = Usage.model_validate(
            {
                "prompt_tokens": completion_response["usage"]["prompt_tokens"] or 0,
                "completion_tokens": completion_response["usage"]["completion_tokens"] or 0,
                "total_tokens": completion_response["usage"]["total_tokens"] or 0,
            }
        )
        usage: Final = Usage(
            prompt_tokens=validated_counts.prompt_tokens,
            completion_tokens=validated_counts.completion_tokens,
            total_tokens=validated_counts.total_tokens,
        )
        setattr(model_response, "usage", usage)
        return model_response

    def validate_environment(
        self,
        headers: Mapping[str, object],
        model: str,
        messages: list[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
        }
        if api_key is not None:
            headers["Authorization"] = f"Token {api_key}"
        return headers
