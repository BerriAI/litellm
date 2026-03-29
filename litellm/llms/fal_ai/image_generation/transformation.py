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


def _normalize_model_endpoint(model: str) -> str:
    normalized = model.strip()
    if normalized.startswith("fal_ai/"):
        return normalized[len("fal_ai/") :]
    return normalized


def _get_model_info_for_image_size_mapping(model: str) -> Optional[dict]:
    import litellm
    from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap

    normalized_model = _normalize_model_endpoint(model)
    candidate_models = [normalized_model]

    if normalized_model.startswith("fal-ai/"):
        candidate_models.append(normalized_model[len("fal-ai/") :])

    for candidate_model in candidate_models:
        model_info = litellm.model_cost.get(candidate_model)
        if (
            model_info is not None
            and "supports_raw_image_size_dimensions" in model_info
        ):
            return model_info

    model_cost_map = GetModelCostMap.load_local_model_cost_map()
    for candidate_model in candidate_models:
        if candidate_model in model_cost_map:
            return model_cost_map[candidate_model]

    return None


def _supports_raw_image_size_dimensions(model: str) -> bool:
    model_info = _get_model_info_for_image_size_mapping(model)
    if model_info is None:
        return False
    return bool(model_info.get("supports_raw_image_size_dimensions"))


def _map_openai_size_to_model_image_size(model: str, size: Any) -> Any:
    if _supports_raw_image_size_dimensions(model):
        if isinstance(size, dict):
            width = size.get("width")
            height = size.get("height")
            if isinstance(width, int) and isinstance(height, int):
                return f"{width}x{height}"
        return size

    return _map_openai_size_to_image_size(size)


def _map_openai_size_to_image_size(size: Any) -> Any:
    if isinstance(size, dict):
        return size

    if not isinstance(size, str):
        return size

    openai_size_to_image_size = {
        "1024x1024": "square_hd",
        "512x512": "square",
        "1792x1024": "landscape_16_9",
        "1024x1792": "portrait_16_9",
        "1024x768": "landscape_4_3",
        "768x1024": "portrait_4_3",
        "1536x1024": "landscape_4_3",
        "1024x1536": "portrait_4_3",
    }
    if size in openai_size_to_image_size:
        return openai_size_to_image_size[size]

    if "x" in size:
        try:
            width_str, height_str = size.split("x")
            return {
                "width": int(width_str),
                "height": int(height_str),
            }
        except (AttributeError, ValueError):
            return size

    return size


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
        Get the complete url for the request.

        Newer Fal AI image generation models are addressed by their model path
        directly under the base URL, e.g. `https://fal.run/fal-ai/flux-2`.
        """
        complete_url: str = (
            api_base or get_secret_str("FAL_AI_API_BASE") or self.DEFAULT_BASE_URL
        )

        complete_url = complete_url.rstrip("/")
        endpoint = self.IMAGE_GENERATION_ENDPOINT or _normalize_model_endpoint(model)
        if endpoint:
            complete_url = f"{complete_url}/{endpoint.lstrip('/')}"
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
        Get supported OpenAI parameters for fal.ai image generation
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
        supported_params = self.get_supported_openai_params(model)
        for key, value in non_default_params.items():
            if key in optional_params:
                continue

            if key not in supported_params:
                if drop_params:
                    continue
                raise ValueError(
                    f"Parameter {key} is not supported for model {model}. Supported parameters are {supported_params}. Set drop_params=True to drop unsupported parameters."
                )

            if key == "n":
                optional_params["num_images"] = value
            elif key == "response_format":
                output_format = "png" if value in ["url", "b64_json"] else value
                optional_params["output_format"] = output_format
            elif key == "size":
                optional_params["image_size"] = _map_openai_size_to_model_image_size(
                    model=model, size=value
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
        """
        Transform the image generation request to the fal.ai image generation request body
        """
        fal_ai_image_generation_request_body = {
            "prompt": prompt,
            **optional_params,
        }
        return fal_ai_image_generation_request_body
