from typing import TYPE_CHECKING, Any, List, Optional

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


class ModelsLabImageGenerationConfig(BaseImageGenerationConfig):
    DEFAULT_BASE_URL: str = "https://modelslab.com/api/v6"
    IMAGE_GENERATION_ENDPOINT: str = "images/text2img"

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        return [
            "n",
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
            if k not in optional_params.keys():
                if k in supported_params:
                    if k == "n":
                        optional_params["samples"] = non_default_params[k]
                    elif k == "size":
                        size_str = non_default_params[k]
                        if "x" in str(size_str):
                            w, h = size_str.split("x", 1)
                            optional_params["width"] = int(w)
                            optional_params["height"] = int(h)
                    else:
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
        complete_url: str = (
            api_base
            or get_secret_str("MODELSLAB_API_BASE")
            or self.DEFAULT_BASE_URL
        )
        complete_url = complete_url.rstrip("/")
        complete_url = f"{complete_url}/{self.IMAGE_GENERATION_ENDPOINT}"
        return complete_url

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
            api_key or get_secret_str("MODELSLAB_API_KEY")
        )
        if not final_api_key:
            raise ValueError(
                "MODELSLAB_API_KEY is not set. Please set the MODELSLAB_API_KEY "
                "environment variable or pass api_key."
            )
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
        # API key goes in the request body, not in headers
        api_key = litellm_params.get("api_key") or get_secret_str("MODELSLAB_API_KEY")

        # Strip provider prefix from model name for model_id
        model_id = model
        if "/" in model_id:
            model_id = model_id.split("/", 1)[1]

        request_body: dict = {
            "key": api_key,
            "prompt": prompt,
            "model_id": model_id,
            **optional_params,
        }
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
                error_message=f"Error parsing ModelsLab response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        status = response_data.get("status", "")

        if status == "error":
            error_message = response_data.get("message", "Unknown error from ModelsLab")
            raise self.get_error_class(
                error_message=f"ModelsLab error: {error_message}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        if status == "processing":
            fetch_url = response_data.get("fetch_result", "")
            eta = response_data.get("eta", "unknown")
            msg = response_data.get("message", "Image is still processing")
            raise self.get_error_class(
                error_message=(
                    f"ModelsLab image is still processing. ETA: {eta}s. "
                    f"Fetch result at: {fetch_url}. Message: {msg}"
                ),
                status_code=202,
                headers=raw_response.headers,
            )

        # status == "success"
        if not model_response.data:
            model_response.data = []

        output_urls = response_data.get("output", [])
        for url in output_urls:
            image_obj = ImageObject(url=url)
            model_response.data.append(image_obj)

        return model_response
