import base64
from typing import TYPE_CHECKING, Any, List, Optional

import httpx

from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.llms.cloudflare.chat.transformation import CloudflareError
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


class CloudflareImageGenerationConfig(BaseImageGenerationConfig):
    """
    Reference: https://developers.cloudflare.com/workers-ai/models/#text-to-image
    """

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        if api_base is None:
            account_id = get_secret_str("CLOUDFLARE_ACCOUNT_ID")
            api_base = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/"
        return api_base + model

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
        if api_key is None:
            raise ValueError(
                "Missing Cloudflare API Key - A call is being made to cloudflare but no key is set either in the environment variables or via params"
            )
        headers["accept"] = "application/json"
        headers["content-type"] = "application/json"
        headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        return ["n", "size"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)
        for k, v in non_default_params.items():
            if k == "size" and isinstance(v, str) and "x" in v:
                parts = v.split("x")
                optional_params["width"] = int(parts[0])
                optional_params["height"] = int(parts[1])
            elif k == "n":
                optional_params["num_images"] = v
            elif k in supported_params:
                optional_params[k] = v
            elif not drop_params:
                raise ValueError(
                    f"Parameter {k} is not supported for model {model}. Supported parameters are {supported_params}. Set drop_params=True to drop unsupported parameters."
                )
        return optional_params

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        return {
            "prompt": prompt,
            **optional_params,
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
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ImageResponse:
        content_type = raw_response.headers.get("content-type", "")

        if not model_response.data:
            model_response.data = []

        # Cloudflare may return raw image bytes (e.g. PNG) or JSON
        if "image/" in content_type:
            image_b64 = base64.b64encode(raw_response.content).decode("utf-8")
            model_response.data.append(
                ImageObject(b64_json=image_b64)
            )
        else:
            try:
                response_data = raw_response.json()
            except Exception as e:
                raise CloudflareError(
                    message=f"Error transforming image generation response: {e}",
                    status_code=raw_response.status_code,
                )

            result = response_data.get("result", {})
            image_data = result.get("image")
            if image_data:
                model_response.data.append(
                    ImageObject(b64_json=image_data)
                )

        return model_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: dict
    ) -> CloudflareError:
        return CloudflareError(
            status_code=status_code,
            message=error_message,
        )
