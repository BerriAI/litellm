from typing import TYPE_CHECKING, Any, List, Optional

import httpx

from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageObject, ImageResponse

from .transformation import FalAIBaseConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class FalAIIdeogramV3Config(FalAIBaseConfig):
    """
    Configuration for fal-ai/ideogram/v3 image generation.

    The Ideogram v3 endpoint exposes multiple generation modes (text-to-image,
    remixing, reframing, background replacement, character workflows, etc.).
    LiteLLM focuses on the text-to-image interface to maintain OpenAI parity.

    Model endpoint: fal-ai/ideogram/v3
    Documentation: https://fal.ai/models/fal-ai/ideogram/v3
    """

    IMAGE_GENERATION_ENDPOINT: str = "fal-ai/ideogram/v3"

    _OPENAI_SIZE_TO_IMAGE_SIZE = {
        "1024x1024": "square_hd",
        "512x512": "square",
        "1024x768": "landscape_4_3",
        "768x1024": "portrait_4_3",
        "1536x1024": "landscape_16_9",
        "1024x1536": "portrait_16_9",
    }

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        """
        Ideogram v3 accepts the core OpenAI image parameters.
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
        """
        Map OpenAI-style parameters onto Ideogram's request schema.
        """

        supported_params = self.get_supported_openai_params(model)

        for k in non_default_params.keys():
            if k in optional_params:
                continue

            if k not in supported_params:
                if drop_params:
                    continue
                raise ValueError(
                    f"Parameter {k} is not supported for model {model}. "
                    f"Supported parameters are {supported_params}. "
                    "Set drop_params=True to drop unsupported parameters."
                )

            value = non_default_params[k]

            if k == "n":
                optional_params["num_images"] = value
            elif k == "size":
                optional_params["image_size"] = self._map_image_size(value)
            elif k == "response_format":
                # Ideogram always returns URLs; nothing to map but don't error.
                continue

        return optional_params

    def _map_image_size(self, size: Any) -> Any:
        if isinstance(size, dict):
            width = size.get("width")
            height = size.get("height")
            if isinstance(width, int) and isinstance(height, int):
                return {"width": width, "height": height}
            return size

        if not isinstance(size, str):
            return size

        normalized = size.strip()
        if normalized in self._OPENAI_SIZE_TO_IMAGE_SIZE:
            return self._OPENAI_SIZE_TO_IMAGE_SIZE[normalized]

        if "x" in normalized:
            try:
                width_str, height_str = normalized.split("x")
                width = int(width_str)
                height = int(height_str)
                return {"width": width, "height": height}
            except (ValueError, AttributeError):
                pass

        # Fallback to a safe default that Ideogram accepts.
        return "square_hd"

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Construct the request payload for Ideogram v3.

        Required:
            - prompt: text prompt describing the scene.

        Optional (subset):
            - rendering_speed, style_preset, style, style_codes, color_palette,
              image_urls, style_reference_images, expand_prompt, seed,
              negative_prompt, image_size, etc.
        """

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
        """
        Parse Ideogram v3 responses which contain a list of File objects.
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

        images = response_data.get("images", [])
        if isinstance(images, list):
            for image_entry in images:
                if isinstance(image_entry, dict):
                    url = image_entry.get("url")
                else:
                    url = image_entry

                model_response.data.append(
                    ImageObject(
                        url=url,
                        b64_json=None,
                    )
                )

        if hasattr(model_response, "_hidden_params") and "seed" in response_data:
            model_response._hidden_params["seed"] = response_data["seed"]

        return model_response


