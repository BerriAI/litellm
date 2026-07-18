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

DEFAULT_API_BASE = "https://api.novita.ai"
CHAT_BASE_SUFFIXES = ("/v3/openai", "/openai")


class NovitaImageGenerationConfig(BaseImageGenerationConfig):
    """
    Configuration for Novita AI Seedream image generation.

    Synchronous per-model endpoint POST {api_base}/v3/{model} that returns
    {"images": ["<url>", ...]}
    https://novita.ai/docs/api-reference/model-apis-seedream-4-0
    """

    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        return ["n", "size"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        passthrough = {k: v for k, v in non_default_params.items() if k not in optional_params and k != "n"}
        n = non_default_params.get("n")
        sequential = {"sequential_image_generation": "auto", "max_images": n} if isinstance(n, int) and n > 1 else {}
        return {**optional_params, **passthrough, **sequential}

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        base_url = (api_base or get_secret_str("NOVITA_API_BASE") or DEFAULT_API_BASE).rstrip("/")
        for suffix in CHAT_BASE_SUFFIXES:
            if base_url.endswith(suffix):
                base_url = base_url[: -len(suffix)]
                break
        return f"{base_url}/v3/{model}"

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
        final_api_key = api_key or get_secret_str("NOVITA_API_KEY")
        if not final_api_key:
            raise ValueError("NOVITA_API_KEY is not set")
        headers["Authorization"] = f"Bearer {final_api_key}"
        headers["Content-Type"] = "application/json"
        headers["X-Novita-Source"] = "litellm"
        return headers

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        return {"prompt": prompt, **optional_params}

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
                error_message=f"Failed to parse Novita image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        images = response_data.get("images")
        if not images:
            raise self.get_error_class(
                error_message=f"Novita image generation response missing images: {response_data}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        model_response.data = [ImageObject(url=self._extract_url(item)) for item in images]
        return model_response

    def _extract_url(self, item: object) -> str | None:
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            value = item.get("image_url") or item.get("url")
            return value if isinstance(value, str) else None
        return None
