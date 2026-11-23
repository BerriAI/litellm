"""
Support for OpenAI's `/v1/images/generations` endpoint on deAPI, with the real per-request cost in `usage.cost`.

Docs: https://docs.deapi.ai
"""

from typing import TYPE_CHECKING, Final

import httpx
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from litellm.litellm_core_utils.core_helpers import keyword_arguments, set_response_cost_in_hidden_params
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_generation.transformation import BaseImageGenerationConfig
from litellm.types.llms.openai import AllMessageValues, OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageResponse

from ..common_utils import DeapiException, endpoint_url, json_headers, pick, reported_cost

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding

_DECODED_JSON: Final = TypeAdapter(object)


class _ImageResponseFields(TypedDict, total=False, extra_items=object):
    pass


_SUPPORTED_PARAMS: Final[tuple[OpenAIImageGenerationOptionalParams, ...]] = (
    "background",
    "n",
    "output_compression",
    "output_format",
    "quality",
    "response_format",
    "size",
    "user",
)


class DeapiImageGenerationConfig(BaseImageGenerationConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIImageGenerationOptionalParams]:  # mutable-ok: inherited contract
        return list(_SUPPORTED_PARAMS)

    def map_openai_params(
        self,
        non_default_params: dict[str, object],  # mutable-ok: inherited contract
        optional_params: dict[str, object],  # mutable-ok: inherited contract
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: inherited contract
        return {**optional_params, **pick(non_default_params, _SUPPORTED_PARAMS)}

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
        return {"model": model, "prompt": prompt, **optional_params}

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict[str, object],  # mutable-ok: inherited contract
        optional_params: dict[str, object],  # mutable-ok: inherited contract
        litellm_params: dict[str, object],  # mutable-ok: inherited contract
        encoding: "Encoding | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        body: Final = _DECODED_JSON.validate_python(raw_response.json())
        response: Final = ImageResponse(**_ImageResponseFields(**keyword_arguments(body, ImageResponse)))
        set_response_cost_in_hidden_params(response, reported_cost(raw_response))
        return response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, object] | httpx.Headers,  # mutable-ok: inherited contract
    ) -> BaseLLMException:
        return DeapiException(message=error_message, status_code=status_code, headers=headers)
