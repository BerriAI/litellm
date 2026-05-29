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
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ImageObject, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

HUNYUAN_BASE_URL = "https://api.cloudai.tencent.com"
HUNYUAN_SUBMIT_ENDPOINT = "v1/aiart/openai/image/submit"
HUNYUAN_QUERY_ENDPOINT = "v1/aiart/openai/image/query"

# Keys owned by litellm that must NOT be forwarded to the Hunyuan API.
_LITELLM_RESERVED_PARAM_KEYS: frozenset = frozenset(
    GenericLiteLLMParams.model_fields.keys()
) | {"model", "extra_body", "extra_headers", "drop_params"}


def extract_hunyuan_extra_params(litellm_params: Dict) -> Dict:
    """Return provider-specific params from litellm_params (non-litellm reserved keys)."""
    return {
        k: v for k, v in litellm_params.items() if k not in _LITELLM_RESERVED_PARAM_KEYS
    }


class HunyuanImageGenerationConfig(BaseImageGenerationConfig):
    """
    Configuration for Tencent Hunyuan image generation via OpenAI-compatible API.

    Submit: POST https://api.cloudai.tencent.com/v1/aiart/openai/image/submit
    Query:  POST https://api.cloudai.tencent.com/v1/aiart/openai/image/query

    HTTP requests and polling are handled by handler.py.
    This class only handles data transformation.
    """

    DEFAULT_BASE_URL: str = HUNYUAN_BASE_URL

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        base = api_base or get_secret_str("HUNYUAN_API_BASE") or HUNYUAN_BASE_URL
        base = base.rstrip("/")
        return f"{base}/{HUNYUAN_SUBMIT_ENDPOINT}"

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
        final_api_key: Optional[str] = api_key or get_secret_str("HUNYUAN_API_KEY")
        if not final_api_key:
            raise ValueError("HUNYUAN_API_KEY is not set")

        headers["Authorization"] = final_api_key
        headers["Content-Type"] = "application/json"
        return headers

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        return [
            "n",
            "quality",
            "response_format",
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
                        f"Supported parameters are {supported_params}. "
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
            "model": model or "gpt-image-2",
        }
        for k, v in optional_params.items():
            request_body[k] = v
        return request_body

    @staticmethod
    def _check_task_status(response_data: Dict[str, Any]) -> str:
        status = response_data.get("status", "").upper()
        verbose_logger.debug(f"Hunyuan task status: {status}")
        if status == "DONE":
            return "done"
        elif status == "FAIL":
            raise ValueError(
                f"Hunyuan image generation failed: {response_data.get('message', 'Unknown error')}"
            )
        elif status in ("WAIT", "RUN"):
            return "running"
        else:
            raise ValueError(f"Unknown Hunyuan task status: {status}")

    @staticmethod
    def _transform_response_to_openai(
        response_data: Dict[str, Any],
        model_response: ImageResponse,
    ) -> ImageResponse:
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
                error_message=f"Error parsing Hunyuan response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        model_response.created = int(time.time())
        return self._transform_response_to_openai(
            response_data=response_data,
            model_response=model_response,
        )

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
