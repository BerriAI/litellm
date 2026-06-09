from typing import TYPE_CHECKING, Any, List, Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageResponse
from litellm.utils import convert_to_model_response_object

from ..common_utils import (
    CometAPIException,
    get_cometapi_complete_url,
    require_cometapi_api_key,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class CometAPIImageGenerationConfig(BaseImageGenerationConfig):
    @staticmethod
    def _normalize_image_usage(response_data: dict) -> None:
        usage = response_data.get("usage")
        if not isinstance(usage, dict):
            return

        if usage.get("input_tokens") is None:
            usage["input_tokens"] = 0
        if usage.get("output_tokens") is None:
            usage["output_tokens"] = 0
        if usage.get("total_tokens") is None:
            usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]

        input_tokens_details = usage.get("input_tokens_details")
        if not isinstance(input_tokens_details, dict):
            usage["input_tokens_details"] = {"image_tokens": 0, "text_tokens": 0}
            return

        if input_tokens_details.get("image_tokens") is None:
            input_tokens_details["image_tokens"] = 0
        if input_tokens_details.get("text_tokens") is None:
            input_tokens_details["text_tokens"] = 0

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        """
        https://api.cometapi.com/v1/images/generations
        """
        return [
            "background",
            "moderation",
            "n",
            "output_compression",
            "output_format",
            "quality",
            "response_format",
            "size",
            "style",
            "user",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)

        for k in non_default_params.keys():
            if k not in optional_params.keys():
                if k in supported_params:
                    optional_params[k] = non_default_params[k]
                elif drop_params:
                    pass
                else:
                    raise ValueError(
                        f"Parameter {k} is not supported for model {model}. Supported parameters are {supported_params}. Set drop_params=True to drop unsupported parameters."
                    )

        return optional_params

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the complete url for the request
        """
        return get_cometapi_complete_url(
            api_base, "images/generations", api_key=api_key
        )

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
        final_api_key = require_cometapi_api_key(api_key)

        headers["Authorization"] = f"Bearer {final_api_key}"
        headers["Content-Type"] = "application/json"
        return headers

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the image generation request to the CometAPI image generation request body

        https://api.cometapi.com/v1/images/generations
        """
        request_body = {
            "prompt": prompt,
            "model": model,
            **optional_params,
        }
        return request_body

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ImageResponse:
        """
        Transform the image generation response to the litellm image response

        https://api.cometapi.com/v1/images/generations
        """
        try:
            response_data = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Error transforming image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        if raw_response.status_code >= 400:
            error_data = response_data.get("error", response_data)
            error_message = (
                error_data.get("message")
                if isinstance(error_data, dict)
                else str(error_data)
            )
            raise self.get_error_class(
                error_message=error_message or str(response_data),
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        self._normalize_image_usage(response_data)
        logging_obj.post_call(
            input=request_data.get("prompt", ""),
            api_key=api_key,
            additional_args={"complete_input_dict": request_data},
            original_response=response_data,
        )
        response_data.update(
            {
                key: value
                for key, value in {
                    "size": optional_params.get("size"),
                    "quality": optional_params.get("quality"),
                    "output_format": optional_params.get(
                        "output_format", optional_params.get("response_format")
                    ),
                }.items()
                if value is not None
            }
        )
        image_response: ImageResponse = convert_to_model_response_object(  # type: ignore
            response_object=response_data,
            model_response_object=model_response,
            response_type="image_generation",
        )

        return image_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return CometAPIException(
            message=error_message, status_code=status_code, headers=headers
        )
