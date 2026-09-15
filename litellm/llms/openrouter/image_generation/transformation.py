"""
OpenRouter Image Generation Support

OpenRouter serves image generation from a dedicated endpoint,
``POST https://openrouter.ai/api/v1/images``. Models whose only output modality is
``image`` (``krea/*``, ``openai/gpt-image-*``, ``bytedance-seed/*``, ...) are reachable
*only* there -- sending them to ``/chat/completions`` answers with an opaque HTTP 500.
Hybrid text+image models such as ``google/gemini-2.5-flash-image`` are served from both,
so this config routes every OpenRouter image generation call through ``/images`` rather
than picking per-model, keeping one code path for all of them. This continues the
approach in upstream BerriAI/litellm#34908; the previous version of this file drove image
generation through ``/chat/completions`` instead, which meant models whose only output
modality is image (e.g. ``openai/gpt-image-2.5-flare``) could never resolve, since that
kind of model has no chat-completions image output at all.

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

from typing import TYPE_CHECKING, Any

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
        """
        Map OpenAI image generation params onto OpenRouter's image API shape.

        Delegates the size -> aspect_ratio conversion and response_format handling to
        image_api.map_image_params so image_edit's map_openai_params stays byte-identical
        in behavior; only the surrounding dict-merge shape differs because the two base
        configs pass params in slightly different argument shapes.
        """
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
        """
        Build the ``/images`` request body: ``{model, prompt, ...mapped optional params}``.

        NON_BODY_PARAMS is excluded because "model" and "prompt" are already set
        explicitly above and "extra_headers" belongs on the transport, not the JSON body.
        """
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
        # Parsing/usage-mapping is shared with image_edit via image_api so a fix to cost
        # or usage extraction only needs to happen once.
        return apply_images_response(
            parsed=parse_images_response(raw_response),
            model_response=model_response,
            model=model,
        )

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        """Get the appropriate error class for OpenRouter errors."""
        return OpenRouterException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
