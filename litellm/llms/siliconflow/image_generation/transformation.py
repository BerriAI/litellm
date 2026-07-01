from typing import TYPE_CHECKING, Optional, cast

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

from ..common_utils import SiliconFlowException, get_dict, get_list

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class SiliconFlowImageGenerationConfig(BaseImageGenerationConfig):
    DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
    IMAGE_GENERATION_ENDPOINT = "images/generations"

    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        return ["n", "size"]

    def map_openai_params(
        self,
        non_default_params: dict[str, object],
        optional_params: dict[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        for param_name, value in non_default_params.items():
            if param_name == "n":
                optional_params["batch_size"] = value
            elif param_name == "size":
                optional_params["image_size"] = value
            elif drop_params:
                continue
            else:
                raise ValueError(
                    "Parameter {} is not supported for model {}. Supported parameters are {}. Set drop_params=True to drop unsupported parameters.".format(
                        param_name,
                        model,
                        self.get_supported_openai_params(model),
                    )
                )
        return optional_params

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        stream: Optional[bool] = None,
    ) -> str:
        complete_url = (
            api_base
            or get_secret_str("SILICONFLOW_API_BASE")
            or self.DEFAULT_BASE_URL
        ).rstrip("/")
        if self.IMAGE_GENERATION_ENDPOINT in complete_url:
            return complete_url
        if complete_url.endswith("/v1"):
            return "{}/{}".format(complete_url, self.IMAGE_GENERATION_ENDPOINT)
        if complete_url == "https://api.siliconflow.cn":
            return "{}/v1/{}".format(complete_url, self.IMAGE_GENERATION_ENDPOINT)
        if "/v1" not in complete_url.split("//", 1)[-1]:
            return "{}/v1/{}".format(complete_url, self.IMAGE_GENERATION_ENDPOINT)
        return "{}/{}".format(complete_url, self.IMAGE_GENERATION_ENDPOINT)

    def validate_environment(
        self,
        headers: dict[str, str],
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict[str, str]:
        final_api_key = api_key or get_secret_str("SILICONFLOW_API_KEY")
        if final_api_key is None:
            raise ValueError("SILICONFLOW_API_KEY is not set")
        return {
            **headers,
            "Authorization": "Bearer {}".format(final_api_key),
            "Content-Type": "application/json",
        }

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        headers: dict[str, str],
    ) -> dict[str, object]:
        return {
            "model": model,
            "prompt": prompt,
            **optional_params,
        }

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict[str, object],
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        encoding: object,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ImageResponse:
        try:
            response_data = get_dict(cast(object, raw_response.json()))
        except Exception:
            raise SiliconFlowException(
                message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        if not model_response.data:
            model_response.data = []
        for image_data in get_list(response_data.get("images", response_data.get("data", []))):
            image_item = get_dict(image_data)
            model_response.data.append(
                ImageObject(
                    b64_json=image_item.get("b64_json"),
                    url=image_item.get("url"),
                )
            )
        return model_response
