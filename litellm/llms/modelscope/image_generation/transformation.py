"""
ModelScope Image Generation Config

Handles transformation between OpenAI-compatible format and ModelScope API format.

API Reference: https://modelscope.cn/docs/model-service/API-Inference/intro
"""

from typing import TYPE_CHECKING, Final

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

    LiteLLMLoggingObj: Final = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj: Final = object


class ModelScopeImageGenerationConfig(BaseImageGenerationConfig):
    """Configuration for ModelScope image generation and editing models."""

    DEFAULT_BASE_URL: Final[str] = "https://api-inference.modelscope.cn/v1"

    @override
    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIImageGenerationOptionalParams]:  # mutable-ok: litellm override signature
        return ["size"]  # mutable-ok: small fixed list

    @override
    def map_openai_params(
        self,
        non_default_params: dict,  # mutable-ok: litellm override signature
        optional_params: dict,  # mutable-ok: litellm override signature
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: litellm override signature
        supported_params: Final = self.get_supported_openai_params(model)
        if drop_params:
            non_default_params = {
                k: v for k, v in non_default_params.items() if k in supported_params
            }  # rebind-ok: filtering in place is the litellm pattern  # mutable-ok: dict comprehension for filtering
        optional_params.update(non_default_params)
        return optional_params

    def _get_base_url(self, api_base: str | None) -> str:
        base_url: Final[str] = api_base or get_secret_str("MODELSCOPE_API_BASE") or self.DEFAULT_BASE_URL
        return base_url.rstrip("/")

    @override
    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,  # mutable-ok: litellm override signature
        litellm_params: dict,  # mutable-ok: litellm override signature
        stream: bool | None = None,
    ) -> str:
        return f"{self._get_base_url(api_base)}/images/generations"

    def get_task_status_url(self, api_base: str | None, task_id: str) -> str:
        return f"{self._get_base_url(api_base)}/tasks/{task_id}"

    @override
    def validate_environment(
        self,
        headers: dict,  # mutable-ok: litellm override signature
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: litellm override signature
        optional_params: dict,  # mutable-ok: litellm override signature
        litellm_params: dict,  # mutable-ok: litellm override signature
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: litellm override signature
        final_api_key: Final[str | None] = api_key or get_secret_str("MODELSCOPE_API_KEY")

        if not final_api_key:
            raise ValueError(
                "MODELSCOPE_API_KEY is not set. Please set it via environment variable or pass api_key parameter."
            )

        default_headers: Final = {  # mutable-ok: http header dict
            "Content-Type": "application/json",
            "Authorization": f"Bearer {final_api_key}",
            ASYNC_MODE_HEADER: "true",
        }

        return {**headers, **default_headers}  # mutable-ok: merged header dict

    def get_polling_headers(
        self,
        headers: dict,  # mutable-ok: caller passes a mutable dict
    ) -> dict:  # mutable-ok: builds a new dict for polling
        return {  # mutable-ok: polling header dict
            "Authorization": headers.get("Authorization", ""),
            TASK_TYPE_HEADER: IMAGE_GENERATION_TASK_TYPE,
        }

    @override
    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,  # mutable-ok: litellm override signature
        litellm_params: dict,  # mutable-ok: litellm override signature
        headers: dict,  # mutable-ok: litellm override signature
    ) -> dict:  # mutable-ok: litellm override signature
        request_data: Final = {  # mutable-ok: request body dict
            "model": model,
            "prompt": prompt,
        }

        extra_body: Final = optional_params.get("extra_body") or {}  # mutable-ok: provider fields dict
        for key, value in {**optional_params, **extra_body}.items():  # mutable-ok: merged params dict
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
        request_data: dict,  # mutable-ok: litellm override signature
        optional_params: dict,  # mutable-ok: litellm override signature
        litellm_params: dict,  # mutable-ok: litellm override signature
        encoding: object,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        try:
            response_data: Final = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Error parsing ModelScope response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        if "errors" in response_data:
            errors: Final = response_data["errors"]
            error_msg: Final = errors.get("message", str(errors)) if isinstance(errors, dict) else str(errors)
            raise self.get_error_class(
                error_message=f"ModelScope error: {error_msg}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        task_status: Final = response_data.get("task_status")
        if task_status == TASK_STATUS_FAILED:
            raise self.get_error_class(
                error_message="ModelScope image generation task failed",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        if task_status != TASK_STATUS_SUCCEED:
            raise self.get_error_class(
                error_message=f"ModelScope task did not succeed: status={task_status}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        output_images: Final = response_data.get("output_images", []) or []  # mutable-ok: API response list

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
        headers: dict | httpx.Headers,  # mutable-ok: litellm override signature
    ) -> BaseLLMException:
        return ModelScopeError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
