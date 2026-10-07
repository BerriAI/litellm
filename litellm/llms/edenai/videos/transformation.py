"""
Support for OpenAI's `/v1/videos` API on Eden AI, served at `/v3/videos`. A job is created, polled and
downloaded through the OpenAI routes; Eden reports `cost` as 0 on the create response and the settled
amount on the status read once the job completes or fails.

Docs: https://www.edenai.co/docs/v3/llms/video-generation
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

import httpx
from httpx._types import RequestFiles

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.videos.transformation import OpenAIVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoObject

from ..common_utils import EdenAIException, authorized_headers, endpoint_url, reported_cost

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


def _usage_with_reported_cost(
    usage: Mapping[str, object] | None, body: bytes
) -> dict[str, object]:  # mutable-ok: VideoObject.usage is a plain dict field
    cost: Final = reported_cost(body)
    return {  # mutable-ok: VideoObject.usage is a plain dict field
        key: value
        for key, value in (*(usage.items() if usage else ()), ("provider_reported_cost_usd", cost))
        if value is not None
    }


class EdenAIVideoConfig(OpenAIVideoConfig):
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

    def use_multipart_form_data(self) -> bool:
        return False

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: dict[str, object],  # mutable-ok: inherited contract
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, object],  # mutable-ok: inherited contract
    ) -> tuple[dict[str, object], RequestFiles, str]:  # mutable-ok: inherited contract
        """A reference image is a multipart file part, or a JSON `{"file_id"}` / `{"image_url"}` object."""
        reference: Final = video_create_optional_request_params.get("input_reference")
        if not isinstance(reference, Mapping):
            return super().transform_video_create_request(
                model=model,
                prompt=prompt,
                api_base=api_base,
                video_create_optional_request_params=video_create_optional_request_params,
                litellm_params=litellm_params,
                headers=headers,
            )
        data, files, url = super().transform_video_create_request(
            model=model,
            prompt=prompt,
            api_base=api_base,
            video_create_optional_request_params={  # mutable-ok: inherited contract
                key: value for key, value in video_create_optional_request_params.items() if key != "input_reference"
            },
            litellm_params=litellm_params,
            headers=headers,
        )
        return {**data, "input_reference": dict(reference)}, files, url  # mutable-ok: JSON body

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
        custom_llm_provider: str | None = None,
        request_data: dict[str, object] | None = None,  # mutable-ok: inherited contract
    ) -> VideoObject:
        video: Final = super().transform_video_create_response(
            model=model,
            raw_response=raw_response,
            logging_obj=logging_obj,
            custom_llm_provider=custom_llm_provider,
            request_data=request_data,
        )
        video.usage = _usage_with_reported_cost(video.usage, raw_response.content)
        return video

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raw_response.raise_for_status()  # the shared GET helpers return error bodies instead of raising
        video: Final = super().transform_video_status_retrieve_response(
            raw_response=raw_response, logging_obj=logging_obj, custom_llm_provider=custom_llm_provider
        )
        video.usage = _usage_with_reported_cost(video.usage, raw_response.content)
        return video

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> bytes:
        raw_response.raise_for_status()  # the shared GET helpers return error bodies instead of raising
        return raw_response.content

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
        custom_llm_provider: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: inherited contract
        raw_response.raise_for_status()  # the shared GET helpers return error bodies instead of raising
        return super().transform_video_list_response(
            raw_response=raw_response, logging_obj=logging_obj, custom_llm_provider=custom_llm_provider
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, object] | httpx.Headers,  # mutable-ok: inherited contract
    ) -> BaseLLMException:
        return EdenAIException(message=error_message, status_code=status_code, headers=headers)
