"""
Support for OpenAI's `/v1/images/generations` endpoint on Eden AI, served at `/v3/images/generations`
for every image model in the catalog with the real per-request cost at the top level of the body.

Docs: https://www.edenai.co/docs/v3/llms/image-generation
"""

from typing import TYPE_CHECKING, Final

import httpx

from litellm.litellm_core_utils.core_helpers import set_response_cost_in_hidden_params
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_generation.transformation import BaseImageGenerationConfig
from litellm.types.llms.openai import AllMessageValues, OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageResponse
from litellm.utils import convert_to_model_response_object

from ..common_utils import EdenAIException, endpoint_url, json_headers, pick, reported_cost

if TYPE_CHECKING:
    import tiktoken

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_SUPPORTED_PARAMS: Final[tuple[OpenAIImageGenerationOptionalParams, ...]] = (
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
)


class EdenAIImageGenerationConfig(BaseImageGenerationConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIImageGenerationOptionalParams]:  # mutable-ok: inherited contract
        return list(_SUPPORTED_PARAMS)  # mutable-ok: inherited contract

    def map_openai_params(
        self,
        non_default_params: dict[str, object],  # mutable-ok: inherited contract
        optional_params: dict[str, object],  # mutable-ok: inherited contract
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: inherited contract
        return {**optional_params, **pick(non_default_params, _SUPPORTED_PARAMS)}  # mutable-ok: inherited contract

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict[str, object],  # mutable-ok: inherited contract
        litellm_params: dict[str, object],  # mutable-ok: inherited contract
        stream: bool | None = None,
    ) -> str:
        return endpoint_url(api_base, "images/generations")

    def validate_environment(
        self,
        headers: dict[str, object],  # mutable-ok: inherited contract
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: inherited contract
        optional_params: dict[str, object],  # mutable-ok: inherited contract
        litellm_params: dict[str, object],  # mutable-ok: inherited contract
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, object]:  # mutable-ok: inherited contract
        return json_headers(headers, api_key, model)

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict[str, object],  # mutable-ok: inherited contract
        litellm_params: dict[str, object],  # mutable-ok: inherited contract
        headers: dict[str, object],  # mutable-ok: inherited contract
    ) -> dict[str, object]:  # mutable-ok: inherited contract
        return {"model": model, "prompt": prompt, **optional_params}  # mutable-ok: inherited contract

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict[str, object],  # mutable-ok: inherited contract
        optional_params: dict[str, object],  # mutable-ok: inherited contract
        litellm_params: dict[str, object],  # mutable-ok: inherited contract
        encoding: "tiktoken.Encoding | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        body: Final = raw_response.json()
        logging_obj.post_call(original_response=body)
        response: Final[ImageResponse] = convert_to_model_response_object(
            response_object=body, model_response_object=model_response, response_type="image_generation"
        )
        set_response_cost_in_hidden_params(response, reported_cost(body))
        return response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, object] | httpx.Headers,  # mutable-ok: inherited contract
    ) -> BaseLLMException:
        return EdenAIException(message=error_message, status_code=status_code, headers=headers)
