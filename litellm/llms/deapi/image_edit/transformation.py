"""
Support for OpenAI's `/v1/images/edits` endpoint on deAPI, with the real per-request cost in `usage.cost`.
deAPI has no inpainting model, so `mask` is not offered.

Docs: https://docs.deapi.ai
"""

from typing import TYPE_CHECKING, Final

import httpx

from litellm.litellm_core_utils.core_helpers import set_response_cost_in_hidden_params
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.image_edit.transformation import OpenAIImageEditConfig
from litellm.utils import ImageResponse

from ..common_utils import DeapiException, authorized_headers, endpoint_url, reported_cost

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_SUPPORTED_PARAMS: Final = (
    "image",
    "prompt",
    "model",
    "n",
    "quality",
    "response_format",
    "size",
    "user",
    "extra_headers",
    "extra_query",
    "extra_body",
    "timeout",
)


class DeapiImageEditConfig(OpenAIImageEditConfig):
    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: inherited contract
        return list(_SUPPORTED_PARAMS)

    def validate_environment(
        self,
        headers: dict[str, object],  # mutable-ok: inherited contract
        model: str,
        api_key: str | None = None,
        litellm_params: dict[str, object] | None = None,  # mutable-ok: inherited contract
        api_base: str | None = None,
    ) -> dict[str, object]:  # mutable-ok: inherited contract
        caller_key: Final[object] = (litellm_params or {}).get("api_key")
        return authorized_headers(headers, api_key or (caller_key if isinstance(caller_key, str) else None), model)

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict[str, object],  # mutable-ok: inherited contract
    ) -> str:
        return endpoint_url(api_base, "images/edits")

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> ImageResponse:
        response: Final = super().transform_image_edit_response(
            model=model, raw_response=raw_response, logging_obj=logging_obj
        )
        set_response_cost_in_hidden_params(response, reported_cost(raw_response))
        return response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, object] | httpx.Headers,  # mutable-ok: inherited contract
    ) -> BaseLLMException:
        return DeapiException(message=error_message, status_code=status_code, headers=headers)
