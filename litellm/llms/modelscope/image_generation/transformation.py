"""
ModelScope Image Generation Config

Handles transformation between OpenAI-compatible format and ModelScope API format.

API Reference: https://modelscope.cn/docs/model-service/API-Inference/intro
"""

from typing import TYPE_CHECKING, Optional, Union

import httpx
from typing_extensions import override

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageObject, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = object


class ModelScopeImageGenerationConfig(BaseImageGenerationConfig):
    """
    Configuration for ModelScope image generation.

    Supports text-to-image models like:
    - Qwen/Qwen-Image-Edit
    - And other ModelScope-hosted image generation models
    """

    DEFAULT_BASE_URL: str = "https://api-inference.modelscope.cn/v1"

    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        """
        Return list of OpenAI params supported by ModelScope.

        ModelScope supports standard OpenAI image generation parameters.
        """
        return [
            "n",  # Number of images to generate
            "size",  # Size of the generated images
            "response_format",  # url or b64_json
            "user",  # User identifier
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI parameters to ModelScope parameters.

        ModelScope uses the same parameter names as OpenAI.
        """
        supported_params = self.get_supported_openai_params(model)
        if drop_params:
            non_default_params = {k: v for k, v in non_default_params.items() if k in supported_params}
        optional_params.update(non_default_params)
        return optional_params

    @override
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
        Get the complete URL for the ModelScope image generation API request.
        """
        base_url: str = api_base or get_secret_str("MODELSCOPE_API_BASE") or self.DEFAULT_BASE_URL
        base_url = base_url.rstrip("/")

        # Return the images endpoint
        return f"{base_url}/images/generations"

    @override
    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Validate environment and set up headers for ModelScope.
        """
        final_api_key: Optional[str] = api_key or get_secret_str("MODELSCOPE_API_KEY")

        if not final_api_key:
            raise ValueError(
                "MODELSCOPE_API_KEY is not set. Please set it via environment variable or pass api_key parameter."
            )

        default_headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {final_api_key}",
        }

        headers = {**headers, **default_headers}
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
        Transform OpenAI-style request to ModelScope request format.

        ModelScope uses the same format as OpenAI for image generation.
        """
        # Build the request body (same as OpenAI)
        request_data: dict = {
            "model": model,
            "prompt": prompt,
        }

        # Add optional params
        for key, value in optional_params.items():
            if key.startswith("_"):
                continue
            request_data[key] = value

        return request_data

    @override
    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: object,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ImageResponse:
        """
        Transform ModelScope response to OpenAI-compatible ImageResponse.

        ModelScope returns the same format as OpenAI:
        {"created": timestamp, "data": [{"url": "..."}]}
        """
        try:
            response_data = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Error parsing ModelScope response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        # Check for errors in response
        if "error" in response_data:
            error_msg = response_data["error"].get("message", str(response_data["error"]))
            raise self.get_error_class(
                error_message=f"ModelScope error: {error_msg}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        # Extract images from response
        data_list = response_data.get("data", [])
        if not model_response.data:
            model_response.data = []

        for item in data_list:
            image_obj = ImageObject(
                url=item.get("url"),
                b64_json=item.get("b64_json"),
                revised_prompt=item.get("revised_prompt"),
            )
            model_response.data.append(image_obj)

        return model_response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],
    ) -> BaseLLMException:
        """Return the appropriate error class for ModelScope."""
        from litellm.exceptions import (
            AuthenticationError,
            BadRequestError,
            InternalServerError,
        )

        if status_code == 400:
            return BadRequestError(  # type: ignore[return-value]
                message=error_message,
                model="",
                llm_provider="modelscope",
            )
        elif status_code == 401:
            return AuthenticationError(  # type: ignore[return-value]
                message=error_message,
                model="",
                llm_provider="modelscope",
            )
        elif status_code >= 500:
            return InternalServerError(  # type: ignore[return-value]
                message=error_message,
                model="",
                llm_provider="modelscope",
            )
        else:
            return BadRequestError(  # type: ignore[return-value]
                message=error_message,
                model="",
                llm_provider="modelscope",
            )
