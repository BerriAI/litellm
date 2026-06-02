"""
Tencent Hunyuan GPT-Maas Image Generation Configuration (Text-to-Image)

API: POST https://tokenhub.tencentmaas.com/v1/aiart/gttext
Auth: Authorization: Bearer <API_KEY>
Response: synchronous, no polling required.

Status field: completed = success, failed = failure.
"""

import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import httpx

from litellm._logging import verbose_logger
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
    LiteLLMLoggingObj = Any

HUNYUAN_GPT_MAAS_BASE_URL = "https://tokenhub.tencentmaas.com"
HUNYUAN_GPT_MAAS_TEXT_ENDPOINT = "v1/aiart/gttext"


class HunyuanGptMaasImageGenerationConfig(BaseImageGenerationConfig):
    """
    Configuration for Tencent Hunyuan GPT-Maas text-to-image generation.

    POST https://tokenhub.tencentmaas.com/v1/aiart/gttext
    Returns synchronously (no polling needed).
    """

    DEFAULT_BASE_URL: str = HUNYUAN_GPT_MAAS_BASE_URL

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        base = (
            api_base
            or get_secret_str("HUNYUAN_GPT_MAAS_API_BASE")
            or HUNYUAN_GPT_MAAS_BASE_URL
        )
        base = base.rstrip("/")
        return f"{base}/{HUNYUAN_GPT_MAAS_TEXT_ENDPOINT}"

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
        final_api_key: Optional[str] = api_key or get_secret_str(
            "HUNYUAN_GPT_MAAS_API_KEY"
        )
        if not final_api_key:
            raise ValueError("HUNYUAN_GPT_MAAS_API_KEY is not set")
        headers["Authorization"] = f"Bearer {final_api_key}"
        headers["Content-Type"] = "application/json"
        return headers

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        return [
            "n",
            "quality",
            "size",
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
            if k not in optional_params:
                if k in supported_params:
                    optional_params[k] = non_default_params[k]
                elif drop_params:
                    pass
                else:
                    raise ValueError(
                        f"Parameter {k} is not supported for model {model}. "
                        f"Supported parameters: {supported_params}. "
                        "Set drop_params=True to drop unsupported parameters."
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
        request_body: Dict[str, Any] = {
            "prompt": prompt,
            "model": model or "custom-textmodel-gt",
        }
        for k, v in optional_params.items():
            request_body[k] = v
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
                error_message=f"Error parsing Hunyuan GPT-Maas response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        status = response_data.get("status", "")
        if status == "failed":
            error_code = response_data.get("error_code", "UnknownError")
            error_message = response_data.get("error_message", "Unknown error")
            raise self.get_error_class(
                error_message=f"Hunyuan GPT-Maas image generation failed [{error_code}]: {error_message}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        verbose_logger.debug(f"Hunyuan GPT-Maas response status: {status}")

        model_response.created = response_data.get("created_at") or int(time.time())
        if not model_response.data:
            model_response.data = []

        for image_item in response_data.get("data", []):
            if isinstance(image_item, dict):
                model_response.data.append(
                    ImageObject(
                        url=image_item.get("url"),
                        b64_json=image_item.get("b64_json"),
                    )
                )
        return model_response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],
    ) -> BaseLLMException:
        return BaseLLMException(
            status_code=status_code,
            message=error_message,
        )
