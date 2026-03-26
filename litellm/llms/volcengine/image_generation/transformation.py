"""
VolcEngine (ByteDance Ark) Image Generation Transformation

Supports Seedream (即梦) and other image generation models via the
Volcengine Ark API: https://www.volcengine.com/docs/6791/1397048

The Ark image generation endpoint is OpenAI-compatible at
POST {api_base}/api/v3/images/generations
"""

from typing import TYPE_CHECKING, Any, List, Optional, Union

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

from ..common_utils import VolcEngineError, get_volcengine_base_url, get_volcengine_headers

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class VolcEngineImageGenerationConfig(BaseImageGenerationConfig):
    """
    Configuration for VolcEngine Ark image generation models (e.g. Seedream / 即梦).

    Reference: https://www.volcengine.com/docs/6791/1397048
    """

    IMAGE_GENERATION_ENDPOINT: str = "api/v3/images/generations"

    # Volcengine-native params that are not in the OpenAI spec but should be
    # forwarded when passed via extra_body or non_default_params.
    VOLCENGINE_EXTRA_PARAMS = (
        "output_format",
        "watermark",
        "guidance_scale",
        "seed",
        "sequential_image_generation",
        "stream",
    )

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        return ["n", "response_format", "size", "quality", "style", "user", "seed"]

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
                        f"Parameter {k} is not supported for model {model}. "
                        f"Supported parameters are {supported_params}. "
                        f"Set drop_params=True to drop unsupported parameters."
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
        base_url = (
            api_base
            or get_secret_str("VOLCENGINE_API_BASE")
            or get_secret_str("ARK_API_BASE")
            or get_volcengine_base_url()
        )
        base_url = base_url.rstrip("/")

        if base_url.endswith("/images/generations"):
            return base_url
        if base_url.endswith("/api/v3"):
            return f"{base_url}/images/generations"
        return f"{base_url}/{self.IMAGE_GENERATION_ENDPOINT}"

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
        final_api_key: Optional[str] = (
            api_key
            or litellm_params.get("api_key")
            or get_secret_str("ARK_API_KEY")
            or get_secret_str("VOLCENGINE_API_KEY")
        )
        if not final_api_key:
            raise ValueError(
                "VolcEngine API key is required. "
                "Set ARK_API_KEY / VOLCENGINE_API_KEY or pass api_key."
            )
        return get_volcengine_headers(api_key=final_api_key, extra_headers=headers)

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        request_body: dict = {
            "model": model,
            "prompt": prompt,
        }
        # Pass through OpenAI-compatible params
        for k in ("n", "size", "response_format", "quality", "style", "user", "seed"):
            if k in optional_params:
                request_body[k] = optional_params[k]

        # Pass through volcengine-native params from extra_body
        extra_body = optional_params.get("extra_body") or {}
        for k in self.VOLCENGINE_EXTRA_PARAMS:
            if k in extra_body:
                request_body[k] = extra_body[k]

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
        try:
            response_data = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Error parsing VolcEngine image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        ## LOGGING
        logging_obj.post_call(
            input=request_data.get("prompt", ""),
            api_key=api_key,
            additional_args={"complete_input_dict": request_data},
            original_response=response_data,
        )

        if not model_response.data:
            model_response.data = []

        for image_data in response_data.get("data", []):
            model_response.data.append(
                ImageObject(
                    url=image_data.get("url"),
                    b64_json=image_data.get("b64_json"),
                )
            )

        model_response.created = response_data.get("created", model_response.created)

        return model_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> VolcEngineError:
        typed_headers: httpx.Headers = (
            headers
            if isinstance(headers, httpx.Headers)
            else httpx.Headers(headers or {})
        )
        return VolcEngineError(
            status_code=status_code,
            message=error_message,
            headers=typed_headers,
        )
