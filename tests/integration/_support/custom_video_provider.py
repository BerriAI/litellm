"""A customer-owned video provider for the proxy's `custom_provider_map`: a job queue with its own wire shape."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import httpx
from httpx._types import RequestFiles

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_llm import CustomLLM
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import encode_video_id_with_provider, extract_original_video_id

_STATUS_BY_STATE: Final = MappingProxyType({"queued": "queued", "running": "in_progress", "done": "completed"})


def _video_object(job: Mapping[str, object], model: str | None, custom_llm_provider: str | None) -> VideoObject:
    job_id: Final = str(job["job_id"])
    seconds: Final = job.get("seconds")
    video: Final = VideoObject(
        id=encode_video_id_with_provider(job_id, custom_llm_provider, model) if custom_llm_provider else job_id,
        object="video",
        status=_STATUS_BY_STATE[str(job["state"])],
        created_at=0,
        model=model,
        seconds=str(seconds) if seconds is not None else None,
    )
    return video


class QueueVideoConfig(BaseVideoConfig):
    def get_supported_openai_params(self, model: str) -> list:  # mutable-ok: BaseVideoConfig contract
        return ["seconds"]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: BaseVideoConfig contract
        return {key: value for key, value in video_create_optional_params.items() if key == "seconds"}

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: BaseVideoConfig contract
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict:  # mutable-ok: BaseVideoConfig contract
        key: Final = (litellm_params.api_key if litellm_params is not None else None) or api_key or ""
        return {**headers, "X-Api-Key": key}

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,  # mutable-ok: BaseVideoConfig contract
    ) -> str:
        if api_base is None:
            raise ValueError("api_base is required")
        return f"{api_base.rstrip('/')}/jobs"

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: dict,  # mutable-ok: BaseVideoConfig contract
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: BaseVideoConfig contract
    ) -> tuple[dict, RequestFiles, str]:  # mutable-ok: BaseVideoConfig contract
        seconds: Final = video_create_optional_request_params.get("seconds")
        return ({"model": model, "prompt": prompt, "seconds": seconds}, [], api_base)

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: dict | None = None,
    ) -> VideoObject:
        return _video_object(raw_response.json(), model, custom_llm_provider)

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: BaseVideoConfig contract
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig contract
        return (f"{api_base}/{extract_original_video_id(video_id)}", {})

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        client: object = None,
    ) -> VideoObject:
        return _video_object(raw_response.json(), None, custom_llm_provider)

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: BaseVideoConfig contract
        variant: str | None = None,
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig contract
        return (f"{api_base}/{extract_original_video_id(video_id)}/file", {})

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        client: object = None,
    ) -> bytes:
        return raw_response.content

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: BaseVideoConfig contract
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: dict[str, object] | None = None,
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig contract
        return (api_base, {})

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseVideoConfig contract
        return raw_response.json()

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: BaseVideoConfig contract
        extra_body: dict[str, object] | None = None,
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig contract
        raise NotImplementedError("the queue backend has no remix")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("the queue backend has no remix")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: BaseVideoConfig contract
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig contract
        raise NotImplementedError("the queue backend has no delete")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        raise NotImplementedError("the queue backend has no delete")


VIDEO_CONFIG: Final = QueueVideoConfig()
IMAGE_ONLY_HANDLER: Final = CustomLLM()
