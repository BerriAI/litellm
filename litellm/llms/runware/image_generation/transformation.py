from typing import TYPE_CHECKING, Any, List, Optional, Union
from uuid import uuid4

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


class RunwareImageGenerationConfig(BaseImageGenerationConfig):
    """
    Configuration for Runware image generation.

    Runware API expects:
    - POST to https://api.runware.ai/v1
    - Request body is an array: [{taskType: "imageInference", ...}]
    - Response: {"data": [{imageURL: "...", ...}]}
    """

    DEFAULT_BASE_URL: str = "https://api.runware.ai/v1"

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        return api_base or get_secret_str("RUNWARE_API_BASE") or self.DEFAULT_BASE_URL

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
        final_api_key: Optional[str] = api_key or get_secret_str("RUNWARE_API_KEY")
        if not final_api_key:
            raise ValueError("RUNWARE_API_KEY is not set")

        headers["Authorization"] = f"Bearer {final_api_key}"
        headers["Content-Type"] = "application/json"
        return headers

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        return [
            "n",
            "size",
            "response_format",
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

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> Union[dict, list]:
        # Parse size into width/height
        width = 1024
        height = 1024
        if "size" in optional_params:
            size = optional_params.pop("size")
            if isinstance(size, str) and "x" in size:
                parts = size.split("x")
                width = int(parts[0])
                height = int(parts[1])

        # Map response_format to outputType
        output_type = "URL"
        if "response_format" in optional_params:
            fmt = optional_params.pop("response_format")
            if fmt == "b64_json":
                output_type = "base64Data"

        # Map n to numberResults
        number_results = 1
        if "n" in optional_params:
            number_results = optional_params.pop("n")

        request_body = {
            "taskType": "imageInference",
            "taskUUID": str(uuid4()),
            "positivePrompt": prompt,
            "model": model,
            "width": width,
            "height": height,
            "numberResults": number_results,
            "outputType": output_type,
            "includeCost": True,
            **optional_params,
        }

        # Runware expects array request body
        return [request_body]

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
                error_message=f"Error transforming image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        if not model_response.data:
            model_response.data = []

        images = response_data.get("data", [])
        for image_data in images:
            if isinstance(image_data, dict):
                model_response.data.append(
                    ImageObject(
                        url=image_data.get("imageURL", None),
                        b64_json=image_data.get("imageBase64Data", None),
                    )
                )

        # Store total cost metadata from Runware response if available
        if images:
            total_cost = sum(
                img.get("cost", 0) or 0
                for img in images
                if isinstance(img, dict) and img.get("cost") is not None
            )
            if total_cost > 0:
                model_response._hidden_params = model_response._hidden_params or {}
                model_response._hidden_params["runware_cost"] = total_cost

        return model_response
