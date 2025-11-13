from typing import TYPE_CHECKING, Any, List, Optional

import httpx

from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageResponse
from litellm.utils import convert_to_model_response_object

if TYPE_CHECKING:
    from litellm.litellm_core_utils.logging import Logging as LiteLLMLoggingObj


class GPTImageGenerationConfig(BaseImageGenerationConfig):
    """
    OpenAI gpt-image-1 image generation config
    """

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        return [
            "background",
            "moderation",
            "n",
            "output_compression",
            "output_format",
            "quality",
            "size",
            "user",
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
            "quality", "high"
        )  # always hd for dall-e-3
        image_response.output_format = optional_params.get(
            "response_format", "png"
        )  # always png for dall-e-3

        return image_response
