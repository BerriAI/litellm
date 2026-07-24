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


class OpenAICompatibleImageGenerationConfig(BaseImageGenerationConfig):
    """
    Generic OpenAI-compatible image generation config.

    Used as the default fallback for models routed through the ``openai/``
    provider that are not OpenAI's own dall-e-* or gpt-image-* models.
    Covers community OpenAI-compatible image endpoints (e.g. third-party
    aggregators and self-hosted services whose /v1/images/generations is
    shaped like OpenAI's).

    Accepts the union of standard OpenAI image-generation params so that the
    same config works whether the upstream follows dall-e-3 semantics
    (``response_format``, ``style``) or gpt-image-1 semantics
    (``background``, ``moderation``, ``output_format``, ``output_compression``).

    Non-standard params passed by the caller (e.g. ``watermark``, ``seed``,
    ``guidance_scale`` on Volcengine ark's doubao-seedream models) are not
    listed here; they are forwarded verbatim to the upstream via ``extra_body``
    by :func:`litellm.utils.add_provider_specific_params_to_optional_params`.
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
            "response_format",
            "size",
            "style",
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

        # passthrough whatever the caller asked for; defaults mirror the
        # dall-e / gpt-image response shape so downstream consumers don't break.
        image_response.size = optional_params.get("size", "1024x1024")
        image_response.quality = optional_params.get("quality")
        image_response.output_format = optional_params.get(
            "output_format", optional_params.get("response_format")
        )

        return image_response
