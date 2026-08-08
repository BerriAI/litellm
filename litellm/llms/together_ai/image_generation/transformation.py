from typing import TYPE_CHECKING, Any, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.llms.together_ai.common_utils import (
    TogetherAIException,
    get_together_ai_images_generations_url,
    map_openai_image_param_to_together_ai,
    resolve_together_ai_api_key,
    together_ai_image_data_to_image_objects,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class TogetherAIImageGenerationConfig(BaseImageGenerationConfig):
    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        return ["n", "size", "response_format", "quality", "style", "user"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        return {
            **optional_params,
            **dict(
                mapped_item
                for key, value in non_default_params.items()
                for mapped_item in map_openai_image_param_to_together_ai(key, value)
            ),
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
        return get_together_ai_images_generations_url(api_base)

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
        resolved_api_key = resolve_together_ai_api_key(api_key)
        if resolved_api_key is None:
            raise TogetherAIException(
                message="Together AI API key is not set. Set TOGETHERAI_API_KEY or pass api_key.",
                status_code=401,
                headers={},
            )
        return {**headers, "Authorization": f"Bearer {resolved_api_key}"}

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        extra_body = optional_params.get("extra_body") or {}
        body_params = {k: v for k, v in optional_params.items() if k not in ("extra_body", "extra_headers")}
        return {"model": model, "prompt": prompt, **body_params, **extra_body}

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
        try:
            response_json = raw_response.json()
        except ValueError as e:
            raise TogetherAIException(
                message=f"Error parsing Together AI image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        model_response.data = together_ai_image_data_to_image_objects(response_json)
        return model_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return TogetherAIException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
