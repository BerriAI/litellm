from typing import TYPE_CHECKING, Any, List, Optional

import httpx

from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams, AllMessageValues
from litellm.types.utils import ImageResponse
from litellm.utils import convert_to_model_response_object

if TYPE_CHECKING:
    from litellm.litellm_core_utils.logging import Logging as LiteLLMLoggingObj


class BurnCloudImageGenerationConfig(BaseImageGenerationConfig):
    def get_supported_openai_params(
            self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        return [
            "n",
            "response_format",
            "style",
            "quality",
            "size",
            "user",
        ]

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
            api_base = get_secret_str("BURNCLOUD_API_BASE")

        # Remove trailing slashes and ensure clean base URL
        api_base = api_base.rstrip("/") if api_base else api_base

        # if endswith "/v1"
        if api_base and api_base.endswith("/v1"):
            api_base = f"{api_base}/images/generations"
        else:
            api_base = f"{api_base}/v1/images/generations"

        return api_base

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
            api_key = get_secret_str("BURNCLOUD_API_KEY")

        default_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        # If 'Authorization' is provided in headers, it overrides the default.
        if "Authorization" in headers:
            default_headers["Authorization"] = headers["Authorization"]

        # Merge other headers, overriding any default ones except Authorization
        return {**default_headers, **headers}

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
                        f"Parameter {k} is not supported for model {model}. Supported parameters are {supported_params}. Set drop_params=True to drop unsupported parameters."
                    )

        return optional_params

    def transform_image_generation_response(
            self,
            model: str,
            raw_response: httpx.Response,
            model_response: ImageResponse,
            logging_obj: "LiteLLMLoggingObj",
            request_data: dict,
            optional_params: dict,
            litellm_params: dict,
            encoding: Any,
            api_key: Optional[str] = None,
            json_mode: Optional[bool] = None,
    ) -> ImageResponse:
        response = raw_response.json()

        stringified_response = response
        ## LOGGING
        logging_obj.post_call(
            input=request_data.get("prompt", ""),
            api_key=api_key,
            additional_args={"complete_input_dict": request_data},
            original_response=stringified_response,
        )
        image_response: ImageResponse = convert_to_model_response_object(  # type: ignore
            response_object=stringified_response,
            model_response_object=model_response,
            response_type="image_generation",
        )

        # set optional params
        image_response.size = optional_params.get(
            "size", "1024x1024"
        )  # default is always 1024x1024
        image_response.quality = optional_params.get(
            "quality", "standard"
        )
        image_response.output_format = optional_params.get(
            "response_format", "url"
        )

        return image_response
