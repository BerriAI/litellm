"""
Support for OpenAI's `/v1/videos` API on deAPI. A job is created, polled and downloaded through the OpenAI
routes, and the status read of a completed job reports the settled cost in `usage.cost`.

Docs: https://docs.deapi.ai
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.videos.transformation import OpenAIVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoObject

from ..common_utils import DeapiException, authorized_headers, endpoint_url, reported_cost

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.custom_httpx.http_handler import HTTPHandler


def _usage_with_reported_cost(
    usage: Mapping[str, object] | None, raw_response: httpx.Response
) -> dict[str, object]:  # mutable-ok: VideoObject.usage is a plain dict field
    cost: Final = reported_cost(raw_response)
    return {
        key: value
        for key, value in (*(usage.items() if usage else ()), ("provider_reported_cost_usd", cost))
        if value is not None
    }


class DeapiVideoConfig(OpenAIVideoConfig):
    def validate_environment(
        self,
        headers: dict[str, object],  # mutable-ok: inherited contract
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict[str, object]:  # mutable-ok: inherited contract
        return authorized_headers(headers, api_key or (litellm_params.api_key if litellm_params else None), model)

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict[str, object],  # mutable-ok: inherited contract
    ) -> str:
        return endpoint_url(api_base, "videos")

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
        custom_llm_provider: str | None = None,
        client: "HTTPHandler | None" = None,
    ) -> VideoObject:
        raw_response.raise_for_status()  # the shared GET helpers return error bodies instead of raising
        video: Final = super().transform_video_status_retrieve_response(
            raw_response=raw_response, logging_obj=logging_obj, custom_llm_provider=custom_llm_provider
        )
        video.usage = _usage_with_reported_cost(video.usage, raw_response)
        return video

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
        client: "HTTPHandler | None" = None,
    ) -> bytes:
        raw_response.raise_for_status()  # the shared GET helpers return error bodies instead of raising
        return raw_response.content

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, object] | httpx.Headers,  # mutable-ok: inherited contract
    ) -> BaseLLMException:
        return DeapiException(message=error_message, status_code=status_code, headers=headers)
