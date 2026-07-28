"""
OpenRouter Image Generation Support

OpenRouter serves image generation from a dedicated endpoint,
``POST https://openrouter.ai/api/v1/images``. Models whose only output modality is
``image`` (``krea/*``, ``openai/gpt-image-*``, ``bytedance-seed/*``, ...) are reachable
*only* there; sending them to ``/chat/completions`` answers with an opaque HTTP 500.
Hybrid text+image models such as ``google/gemini-2.5-flash-image`` are served from both,
so this config routes every OpenRouter image generation through ``/images``.

Request format:
{
    "model": "krea/krea-2-large",
    "prompt": "a red panda astronaut floating in space",
    "n": 1,
    "aspect_ratio": "1:1"
}

See ``litellm/llms/openrouter/image_api.py`` for the response format and the parameter
mapping shared with image edit.
"""

from typing import TYPE_CHECKING, Any, Union

import httpx

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.llms.openrouter.common_utils import OpenRouterException
from litellm.llms.openrouter.image_api import (
    NON_BODY_PARAMS,
    apply_images_response,
    map_image_params,
    parse_images_response,
    resolve_images_url,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class OpenRouterImageGenerationConfig(BaseImageGenerationConfig):
    """Maps ``/v1/images/generations`` onto OpenRouter's ``/api/v1/images`` endpoint."""

    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        return ["n", "quality", "response_format", "size"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        return {
            **optional_params,
            **map_image_params(params=non_default_params, model=model, drop_params=drop_params),
        }

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        return resolve_images_url(api_base)

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
        resolved_api_key = api_key or litellm.api_key or get_secret_str("OPENROUTER_API_KEY")
        headers.update(
            {
                "Authorization": f"Bearer {resolved_api_key}",
            }
        )
        return headers

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        return {
            "model": model,
            "prompt": prompt,
            **{key: value for key, value in optional_params.items() if key not in NON_BODY_PARAMS},
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
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        return apply_images_response(
            parsed=parse_images_response(raw_response),
            model_response=model_response,
            model=model,
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return OpenRouterException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
