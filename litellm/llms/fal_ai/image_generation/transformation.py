from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

import httpx
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

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
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class FalImageProviderSpecificFields(TypedDict, total=False):
    width: ReadOnly[int]
    height: ReadOnly[int]
    content_type: ReadOnly[str]


_FAL_IMAGE_DATA: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])


def fal_images_to_image_objects(images: object) -> tuple[ImageObject, ...]:
    if not isinstance(images, list):
        return ()

    def to_image_object(image_data: object) -> ImageObject:
        if isinstance(image_data, Mapping):
            image_map: Final = _FAL_IMAGE_DATA.validate_python(image_data)
            url: Final = image_map.get("url")
            b64_json: Final = image_map.get("b64_json")
            width: Final = image_map.get("width")
            height: Final = image_map.get("height")
            content_type: Final = image_map.get("content_type")
            provider_specific_fields: Final[FalImageProviderSpecificFields] = {
                **({"width": width} if isinstance(width, int) and type(width) is int and width > 0 else {}),
                **({"height": height} if isinstance(height, int) and type(height) is int and height > 0 else {}),
                **({"content_type": content_type} if isinstance(content_type, str) else {}),
            }
            return ImageObject(
                url=url if isinstance(url, str) else None,
                b64_json=b64_json if isinstance(b64_json, str) else None,
                provider_specific_fields=provider_specific_fields or None,
            )
        return ImageObject(url=image_data if isinstance(image_data, str) else None, b64_json=None)

    return tuple(to_image_object(image_data) for image_data in images if isinstance(image_data, (Mapping, str)))


class FalAIBaseConfig(BaseImageGenerationConfig):
    """
    Base configuration for Fal AI image generation models.
    Handles common functionality like URL construction and authentication.
    """

    DEFAULT_BASE_URL: str = "https://fal.run"
    IMAGE_GENERATION_ENDPOINT: str = ""

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        """
        Get the complete url for the request

        Some providers need `model` in `api_base`
        """
        complete_url: str = api_base or get_secret_str("FAL_AI_API_BASE") or self.DEFAULT_BASE_URL

        complete_url = complete_url.rstrip("/")
        if self.IMAGE_GENERATION_ENDPOINT:
            complete_url = f"{complete_url}/{self.IMAGE_GENERATION_ENDPOINT}"
        return complete_url

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        final_api_key: Final[str | None] = api_key or get_secret_str("FAL_AI_API_KEY")
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
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        """
        Transform the image generation response to the litellm image response
        """
        try:
            response_data: Final = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Error transforming image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        if not model_response.data:
            model_response.data = []

        model_response.data.extend(fal_images_to_image_objects(response_data.get("images", ())))
        return model_response


class FalAIImageGenerationConfig(FalAIBaseConfig):
    """
    Default Fal AI image generation configuration for generic models.
    """

    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
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
        supported_params: Final = self.get_supported_openai_params(model)
        for k in non_default_params:
            if k not in optional_params:
                if k in supported_params:
                    optional_params[k] = non_default_params[k]
                elif drop_params:
                    pass
                else:
                    raise ValueError(
                        f"Parameter {k} is not supported for model {model}. Supported parameters are {supported_params}. Set drop_params=True to drop unsupported parameters."
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
        fal_ai_image_generation_request_body: Final = {
            "prompt": prompt,
            **optional_params,
        }
        return fal_ai_image_generation_request_body
