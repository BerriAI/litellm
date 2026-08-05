"""
MachGen image generation (text-to-image) transformation.

MachGen is task based: `POST /api/v0/generate` returns a `task_id`, the caller polls
`GET /api/v0/tasks/{task_id}` until the task is terminal, then downloads the asset.
Polling lives in the handler; this module only translates data.

API reference: https://www.machgen.ai/docs/rest_api
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import httpx

from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageObject, ImageResponse

from ..common_utils import (
    DEFAULT_API_BASE,
    GENERATE_PATH,
    IMAGE_OUTPUT_KEY,
    MACHGEN_IMAGE_CONFIG_PARAMS,
    MACHGEN_TOP_LEVEL_PARAMS,
    STATUS_COMPLETED,
    STATUS_FAILED,
    TEXT_TO_IMAGE_TASK_TYPE,
    MachGenError,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


DEFAULT_HEIGHT = 1024


class MachGenImageGenerationConfig(BaseImageGenerationConfig):
    """Translate OpenAI image generation requests to/from the MachGen task API."""

    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        return ["size", "response_format"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)
        passthrough_params = MACHGEN_IMAGE_CONFIG_PARAMS | MACHGEN_TOP_LEVEL_PARAMS

        for key, value in non_default_params.items():
            if key in optional_params or value is None:
                continue
            if key == "size":
                width, height = self._parse_size(value)
                optional_params["width"] = width
                optional_params["height"] = height
            elif key in supported_params or key in passthrough_params:
                optional_params[key] = value
            elif not drop_params:
                raise ValueError(
                    f"Parameter {key} is not supported for model {model}. "
                    f"Supported parameters are {sorted(set(supported_params) | passthrough_params)}. "
                    "Set drop_params=True to drop unsupported parameters."
                )

        return optional_params

    @staticmethod
    def _parse_size(size: str) -> tuple[int, int]:
        try:
            width, height = (int(part) for part in size.lower().split("x"))
        except ValueError as e:
            raise ValueError(f"Invalid size format: '{size}'. Expected 'WIDTHxHEIGHT' (e.g. '1024x1024').") from e
        return width, height

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        resolved_api_key = api_key or get_secret_str("MACHGEN_API_KEY")
        if not resolved_api_key:
            raise MachGenError(
                status_code=401,
                message="MACHGEN_API_KEY is not set. Set the environment variable or pass api_key.",
            )

        return {
            **headers,
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json",
        }

    def get_api_base(self, api_base: str | None) -> str:
        return (api_base or get_secret_str("MACHGEN_API_BASE") or DEFAULT_API_BASE).rstrip("/")

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        return f"{self.get_api_base(api_base)}{GENERATE_PATH}"

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        image_config = {
            key: optional_params[key] for key in MACHGEN_IMAGE_CONFIG_PARAMS if optional_params.get(key) is not None
        }
        top_level_params = {
            key: optional_params[key] for key in MACHGEN_TOP_LEVEL_PARAMS if optional_params.get(key) is not None
        }

        return {
            "prompt": prompt,
            "model": model,
            "task_type": TEXT_TO_IMAGE_TASK_TYPE,
            "image_config": {"height": DEFAULT_HEIGHT, **image_config},
            **top_level_params,
        }

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
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        model_response.data = [ImageObject(url=self.get_asset_url(raw_response))]
        model_response.created = int(time.time())
        return model_response

    def get_asset_url(self, raw_response: httpx.Response) -> str:
        try:
            task = raw_response.json()
        except ValueError as e:
            raise MachGenError(
                status_code=raw_response.status_code,
                message=f"Error parsing MachGen task response: {e}",
            ) from e

        status = task.get("status")
        if status == STATUS_FAILED:
            raise MachGenError(
                status_code=400,
                message=f"MachGen image generation failed: {task.get('error_msg') or task.get('moderation')}",
            )
        if status != STATUS_COMPLETED:
            raise MachGenError(
                status_code=500,
                message=f"Unexpected MachGen task status: {status}",
            )

        asset_url = (task.get("task_output") or {}).get(IMAGE_OUTPUT_KEY)
        if not asset_url:
            raise MachGenError(
                status_code=500,
                message="MachGen task completed without an image asset",
            )
        return asset_url

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> MachGenError:
        return MachGenError(status_code=status_code, message=error_message, headers=headers)
