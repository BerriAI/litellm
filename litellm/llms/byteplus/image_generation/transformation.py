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

DEFAULT_API_BASE = "https://ark.ap-southeast.bytepluses.com/api/v3"
IMAGE_GENERATION_ENDPOINT = "images/generations"


class BytePlusImageGenerationConfig(BaseImageGenerationConfig):
    """
    Configuration for BytePlus ModelArk (Seedream) image generation.

    OpenAI-compatible POST {api_base}/images/generations
    https://docs.byteplus.com/en/docs/ModelArk/1541523
    """

    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        return ["n", "size", "response_format"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        passthrough = {k: v for k, v in non_default_params.items() if k not in optional_params and k != "n"}
        n = non_default_params.get("n")
        sequential = (
            {
                "sequential_image_generation": "auto",
                "sequential_image_generation_options": {"max_images": n},
            }
            if isinstance(n, int) and n > 1
            else {}
        )
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
        base_url = api_base or get_secret_str("BYTEPLUS_API_BASE") or get_secret_str("ARK_API_BASE") or DEFAULT_API_BASE
        base_url = base_url.rstrip("/")
        if base_url.endswith(IMAGE_GENERATION_ENDPOINT):
            return base_url
        return f"{base_url}/{IMAGE_GENERATION_ENDPOINT}"

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
        final_api_key = api_key or get_secret_str("BYTEPLUS_API_KEY") or get_secret_str("ARK_API_KEY")
        if not final_api_key:
            raise ValueError("BYTEPLUS_API_KEY or ARK_API_KEY is not set")
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
        return {"model": model, "prompt": prompt, **optional_params}

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
                error_message=f"Failed to parse BytePlus image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        if "error" in response_data and "data" not in response_data:
            error = response_data["error"]
            raise self.get_error_class(
                error_message=str(error.get("message", error) if isinstance(error, dict) else error),
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        model_response.data = [
            ImageObject(
                url=item.get("url"),
                b64_json=item.get("b64_json"),
            )
            for item in response_data.get("data", [])
        ]
        return model_response
