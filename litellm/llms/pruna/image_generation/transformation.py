from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

DEFAULT_API_BASE = "https://api.pruna.ai"
PREDICTIONS_ENDPOINT = "v1/predictions"


class PrunaImageGenerationConfig(BaseImageGenerationConfig):
    """
    Configuration for Pruna AI image generation.

    Pruna is not OpenAI-compatible. The model is passed via a `Model` header,
    auth via an `apikey` header, and `Try-Sync: true` returns the result inline
    within 60 seconds instead of an async prediction id
    https://docs.api.pruna.ai/guides/models/p-image
    """

    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        return ["size"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        size = non_default_params.get("size")
        dims = self._size_to_dimensions(size)
        passthrough = {
            k: v for k, v in non_default_params.items() if k not in optional_params and k not in ("n", "size")
        }
        return {**optional_params, **passthrough, **dims}

    def _size_to_dimensions(self, size: object) -> dict:
        if not isinstance(size, str) or "x" not in size:
            return {}
        width, _, height = size.partition("x")
        if not width.isdigit() or not height.isdigit():
            return {}
        return {"width": int(width), "height": int(height), "aspect_ratio": "custom"}

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        base_url = (api_base or get_secret_str("PRUNA_API_BASE") or DEFAULT_API_BASE).rstrip("/")
        if base_url.endswith(PREDICTIONS_ENDPOINT):
            return base_url
        return f"{base_url}/{PREDICTIONS_ENDPOINT}"

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
        final_api_key = api_key or get_secret_str("PRUNA_API_KEY")
        if not final_api_key:
            raise ValueError("PRUNA_API_KEY is not set")
        headers["apikey"] = final_api_key
        headers["Model"] = model
        headers["Try-Sync"] = "true"
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
        return {"input": {"prompt": prompt, **optional_params}}

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
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        if raw_response.status_code != 200:
            raise self.get_error_class(
                error_message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        try:
            response_data = raw_response.json()
        except ValueError as e:
            raise self.get_error_class(
                error_message=f"Failed to parse Pruna image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        generation_url = response_data.get("generation_url")
        if response_data.get("status") != "succeeded" or not generation_url:
            raise self.get_error_class(
                error_message=(f"Pruna synchronous generation did not complete in time; response: {response_data}"),
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        model_response.data = [ImageObject(url=self._absolute_url(raw_response, generation_url))]
        return model_response

    def _absolute_url(self, raw_response: httpx.Response, generation_url: str) -> str:
        if generation_url.startswith("http"):
            return generation_url
        request_url = raw_response.request.url
        return f"{request_url.scheme}://{request_url.host}{generation_url}"
