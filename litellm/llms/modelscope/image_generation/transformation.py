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
from litellm.llms.modelscope.common_utils import (
    ASYNC_MODE_HEADER,
    IMAGE_GENERATION_TASK_TYPE,
    TASK_STATUS_FAILED,
    TASK_STATUS_SUCCEED,
    TASK_TYPE_HEADER,
    ModelScopeError,
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
    """Configuration for ModelScope image generation and editing models."""

    DEFAULT_BASE_URL: str = "https://api-inference.modelscope.cn/v1"

    @override
    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        # Only size is honored (verified via real API: n is silently ignored,
        # response_format/user are not honored by ModelScope).
        # Provider-specific fields (negative_prompt, seed, steps, guidance,
        # loras, image_url) pass through via extra_body, not this list.
        return ["size"]

    @override
    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)
        if drop_params:
            non_default_params = {k: v for k, v in non_default_params.items() if k in supported_params}
        optional_params.update(non_default_params)
        return optional_params

    def _get_base_url(self, api_base: Optional[str]) -> str:
        base_url: str = api_base or get_secret_str("MODELSCOPE_API_BASE") or self.DEFAULT_BASE_URL
        return base_url.rstrip("/")

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
        return f"{self._get_base_url(api_base)}/images/generations"

    def get_task_status_url(self, api_base: Optional[str], task_id: str) -> str:
        """Build the URL used to poll an async image generation task."""
        return f"{self._get_base_url(api_base)}/tasks/{task_id}"

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
            # ModelScope image-gen only supports async mode: the submit call
            # returns a task_id and the caller must poll GET /v1/tasks/{task_id}.
            ASYNC_MODE_HEADER: "true",
        }

        headers = {**headers, **default_headers}
        return headers

    def get_polling_headers(self, headers: dict) -> dict:
        return {
            "Authorization": headers.get("Authorization", ""),
            TASK_TYPE_HEADER: IMAGE_GENERATION_TASK_TYPE,
        }

    @override
    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        request_data: dict = {
            "model": model,
            "prompt": prompt,
        }

        # litellm wraps provider fields (e.g. image_url) in extra_body; flatten
        # them into the body. Skip extra_headers/extra_query (litellm control
        # params swept in alongside, may carry secrets) and model/prompt (set
        # above from the routed values; a client-controlled extra_body must not
        # override the authorized model or prompt).
        extra_body = optional_params.get("extra_body") or {}
        for key, value in {**optional_params, **extra_body}.items():
            if key in ("extra_body", "extra_headers", "extra_query", "model", "prompt") or key.startswith("_"):
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
        Transform a completed ModelScope task response into an ImageResponse.

        The polled response looks like:
        {"task_status": "SUCCEED", "output_images": ["https://..."], ...}
        """
        try:
            response_data = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Error parsing ModelScope response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        if "errors" in response_data:
            errors = response_data["errors"]
            error_msg = errors.get("message", str(errors)) if isinstance(errors, dict) else str(errors)
            raise self.get_error_class(
                error_message=f"ModelScope error: {error_msg}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        task_status = response_data.get("task_status")
        if task_status == TASK_STATUS_FAILED:
            raise self.get_error_class(
                error_message="ModelScope image generation task failed",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        if task_status != TASK_STATUS_SUCCEED:
            raise self.get_error_class(
                error_message=(f"ModelScope task did not succeed: status={task_status}"),
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        output_images = response_data.get("output_images", []) or []

        for image_url in output_images:
            model_response.data.append(ImageObject(url=image_url))

        if not model_response.data:
            raise self.get_error_class(
                error_message="ModelScope task SUCCEED but no output_images",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        return model_response

    @override
    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],
    ) -> BaseLLMException:
        """Return the ModelScope error class, preserving the real status code."""
        return ModelScopeError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
