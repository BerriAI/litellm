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


class FalAIBaseConfig(BaseImageGenerationConfig):
    """
    Base configuration for Fal AI image generation models.
    Handles common functionality like URL construction and authentication.
    """

    DEFAULT_BASE_URL: str = "https://fal.run"
    IMAGE_GENERATION_ENDPOINT: str = ""

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the complete url for the request

        Some providers need `model` in `api_base`
        """
        complete_url: str = (
            api_base or get_secret_str("FAL_AI_API_BASE") or self.DEFAULT_BASE_URL
        )

        complete_url = complete_url.rstrip("/")
        if self.IMAGE_GENERATION_ENDPOINT:
            complete_url = f"{complete_url}/{self.IMAGE_GENERATION_ENDPOINT}"
        else:
            # Generic models need the model name in the URL path
            # model arrives without provider prefix (e.g. "nano-banana-2/edit")
            # Strip fal-ai/ if already present to avoid double prefix
            model_path = model.removeprefix("fal-ai/")
            complete_url = f"{complete_url}/fal-ai/{model_path}"
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
        final_api_key: Optional[str] = api_key or get_secret_str("FAL_AI_API_KEY")
        if not final_api_key:
            raise ValueError("FAL_AI_API_KEY is not set")

        headers["Authorization"] = f"Key {final_api_key}"
        return headers

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
        """
        Transform the image generation response to the litellm image response
        """
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

        # Preserve requested size on the response for cost calculation
        if optional_params.get("size"):
            model_response.size = optional_params["size"]

        # Handle fal.ai response format
        images = response_data.get("images", [])
        if isinstance(images, list):
            for image_data in images:
                if isinstance(image_data, dict):
                    model_response.data.append(
                        ImageObject(
                            url=image_data.get("url", None),
                            b64_json=image_data.get("b64_json", None),
                        )
                    )
                elif isinstance(image_data, str):
                    # If images is just a list of URLs
                    model_response.data.append(
                        ImageObject(
                            url=image_data,
                            b64_json=None,
                        )
                    )

        return model_response


class FalAIImageGenerationConfig(FalAIBaseConfig):
    """
    Default Fal AI image generation configuration for generic models.
    """

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        """
        Get supported OpenAI-standard parameters for fal.ai image generation.

        IMPORTANT: Only list standard OpenAI params here, NOT FAL-specific params.
        FAL-specific params (image_urls, loras, guidance_scale, etc.) must NOT be
        listed here because LiteLLM's add_provider_specific_params_to_optional_params()
        uses this list as an exclusion filter -- params IN this list are assumed to be
        handled by map_openai_params via _get_non_default_params, but that function
        only processes keys matching default_params (n, quality, size, style, user).
        Unlisted params flow through the provider-specific safety net instead.
        """
        return [
            "n",
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
        """Pass through all params for FAL, flattening extra_body if present.

        LiteLLM's image_generation() path doesn't extract extra_body from kwargs
        (unlike image_edit), so it arrives here as a nested dict. Flatten it so
        FAL-specific params like image_urls reach the FAL API at the top level.
        """
        for k, v in non_default_params.items():
            if k == "extra_body" and isinstance(v, dict):
                for ek, ev in v.items():
                    if ek not in optional_params:
                        optional_params[ek] = ev
            elif k not in optional_params:
                optional_params[k] = v
        return optional_params

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the image generation request to the fal.ai image generation request body.

        Flattens extra_body if present -- LiteLLM's image_generation() path does not
        extract extra_body from kwargs (unlike image_edit), so it can arrive here as
        a nested dict. We flatten it so FAL-specific params (image_urls, loras, etc.)
        appear at the top level of the request body.
        """
        if "extra_body" in optional_params and isinstance(
            optional_params["extra_body"], dict
        ):
            extra = optional_params.pop("extra_body")
            for k, v in extra.items():
                if k not in optional_params:
                    optional_params[k] = v

        fal_ai_image_generation_request_body = {
            "prompt": prompt,
            **optional_params,
        }
        return fal_ai_image_generation_request_body
