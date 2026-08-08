"""MiniMax v1 video generation transformations."""

import base64
from os import PathLike
from typing import TYPE_CHECKING
from urllib.parse import quote, urlsplit, urlunsplit

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.images.utils import ImageEditRequestUtils
from litellm.litellm_core_utils.url_utils import async_safe_get, safe_get
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    encode_video_id_with_provider,
    extract_original_video_id,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = object


class MinimaxVideoConfig(BaseVideoConfig):
    """Configuration for MiniMax's v1 text-to-video and image-to-video API."""

    def get_supported_openai_params(self, model: str) -> list:  # mutable-ok: BaseVideoConfig requires a list.
        return [  # mutable-ok: BaseVideoConfig requires a list result.
            "model",
            "prompt",
            "input_reference",
            "seconds",
            "size",
            "user",
            "extra_headers",
            "extra_body",
            "prompt_optimizer",
            "fast_pretreatment",
            "duration",
            "resolution",
            "callback_url",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: BaseVideoConfig requires a mutable parameter mapping.
        mapped_params: dict = {}  # mutable-ok: Provider parameters are assembled incrementally.

        for key, value in video_create_optional_params.items():
            if value is None or key in ("model", "prompt", "extra_headers", "user"):
                continue
            if key == "input_reference":
                mapped_params["first_frame_image"] = self._prepare_first_frame_image(value)
            elif key == "seconds":
                mapped_params["duration"] = self._coerce_duration(value)
            elif key == "size":
                mapped_params["resolution"] = value
            elif key != "extra_body":
                mapped_params[key] = value

        extra_body = video_create_optional_params.get("extra_body")
        if isinstance(extra_body, dict):
            mapped_params.update(  # mutable-ok: Provider extra fields must merge into the request mapping.
                {  # mutable-ok: Provider extra fields must merge into the request mapping.
                    key: value for key, value in extra_body.items() if value is not None
                }
            )

        return mapped_params

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: BaseVideoConfig passes a mutable header mapping.
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict:  # mutable-ok: BaseVideoConfig requires mutable headers.
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key

        api_key = api_key or litellm.api_key or get_secret_str("MINIMAX_API_KEY")
        if not api_key:
            raise ValueError(
                "MiniMax API key is required. Set MINIMAX_API_KEY environment variable or pass api_key parameter."
            )

        headers.update(
            {  # mutable-ok: Headers are updated in place by the provider interface.
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,  # mutable-ok: BaseVideoConfig defines this parameter as a dict.
    ) -> str:
        """Return the regional MiniMax v1 root used by all video operations."""
        base_url = api_base or get_secret_str("MINIMAX_API_BASE") or "https://api.minimax.io/v1"
        base_url = base_url.rstrip("/")
        for suffix in ("/query/video_generation", "/video_generation", "/files/retrieve"):
            if base_url.endswith(suffix):
                base_url = base_url[: -len(suffix)]
                break
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        return base_url

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: dict,  # mutable-ok: BaseVideoConfig defines mutable request parameters.
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: BaseVideoConfig passes a mutable header mapping.
    ) -> tuple[dict, RequestFiles, str]:  # mutable-ok: BaseVideoConfig requires a dict payload.
        request_data: dict = {  # mutable-ok: Provider request fields are assembled incrementally.
            "model": model,
            "prompt": prompt,
        }
        request_data.update(video_create_optional_request_params)
        request_data.pop("extra_headers", None)
        request_data.pop("extra_body", None)
        request_data.pop("user", None)
        return request_data, [], f"{api_base.rstrip('/')}/video_generation"  # mutable-ok: No multipart files are sent.

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: dict | None = None,  # mutable-ok: BaseVideoConfig supplies request metadata as a dict.
    ) -> VideoObject:
        response_data = self._parse_json_response(raw_response)
        self._raise_for_provider_error(raw_response, response_data)
        task_id = response_data.get("task_id")
        if task_id is None:
            raise ValueError("MiniMax did not return a task_id for video generation")

        video_data: dict = {  # mutable-ok: Optional response metadata is added after validation.
            "id": str(task_id),
            "object": "video",
            "status": self._map_status(response_data.get("status", "queueing")),
            "model": model,
        }
        self._add_request_metadata(video_data, request_data)
        video_obj = VideoObject(**video_data)
        self._wrap_video_id(video_obj, custom_llm_provider, model)
        video_obj.usage = self._usage_from_video(video_obj)
        return video_obj

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: BaseVideoConfig passes a mutable header mapping.
        variant: str | None = None,
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig requires a dict query payload.
        task_id = quote(extract_original_video_id(video_id), safe="")
        return (  # mutable-ok: The provider query has no separate parameter payload.
            f"{api_base.rstrip('/')}/query/video_generation?task_id={task_id}",
            {},  # mutable-ok: The provider query has no separate parameter payload.
        )

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        query_data = self._parse_json_response(raw_response)
        self._raise_for_provider_error(raw_response, query_data)
        file_id = query_data.get("file_id")
        if file_id is None:
            status = self._map_status(query_data.get("status", "processing"))
            raise ValueError(f"MiniMax video is not ready for download (status: {status})")

        headers = self._request_headers(raw_response)
        api_base = self._api_base_from_response(raw_response)
        file_url = f"{api_base}/files/retrieve?file_id={quote(str(file_id), safe='')}"
        client: HTTPHandler = _get_httpx_client()
        file_response = client.get(file_url, headers=headers)
        self._raise_for_status(file_response)
        if self._is_binary_response(file_response):
            return file_response.content

        file_data = self._parse_json_response(file_response)
        self._raise_for_provider_error(file_response, file_data)
        download_url = self._get_download_url(file_data)
        if not download_url:
            raise ValueError("MiniMax did not return a video download URL")

        video_response = safe_get(client, download_url)
        self._raise_for_status(video_response)
        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        query_data = self._parse_json_response(raw_response)
        self._raise_for_provider_error(raw_response, query_data)
        file_id = query_data.get("file_id")
        if file_id is None:
            status = self._map_status(query_data.get("status", "processing"))
            raise ValueError(f"MiniMax video is not ready for download (status: {status})")

        headers = self._request_headers(raw_response)
        api_base = self._api_base_from_response(raw_response)
        file_url = f"{api_base}/files/retrieve?file_id={quote(str(file_id), safe='')}"
        client: AsyncHTTPHandler = get_async_httpx_client(llm_provider=litellm.LlmProviders.MINIMAX)
        file_response = await client.get(file_url, headers=headers)
        self._raise_for_status(file_response)
        if self._is_binary_response(file_response):
            return file_response.content

        file_data = self._parse_json_response(file_response)
        self._raise_for_provider_error(file_response, file_data)
        download_url = self._get_download_url(file_data)
        if not download_url:
            raise ValueError("MiniMax did not return a video download URL")

        video_response = await async_safe_get(client, download_url)
        self._raise_for_status(video_response)
        return video_response.content

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: BaseVideoConfig passes a mutable header mapping.
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig requires a dict query payload.
        task_id = quote(extract_original_video_id(video_id), safe="")
        return (  # mutable-ok: The provider query has no separate parameter payload.
            f"{api_base.rstrip('/')}/query/video_generation?task_id={task_id}",
            {},  # mutable-ok: The provider query has no separate parameter payload.
        )

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        response_data = self._parse_json_response(raw_response)
        self._raise_for_provider_error(raw_response, response_data)
        task_id = response_data.get("task_id")
        if task_id is None:
            raise ValueError("MiniMax did not return a task_id for video status")
        model = response_data.get("model")
        video_data: dict = {  # mutable-ok: Optional status fields are added after validation.
            "id": str(task_id),
            "object": "video",
            "status": self._map_status(response_data.get("status", "processing")),
            "model": model,
        }
        if response_data.get("status") and self._map_status(response_data["status"]) == "failed":
            video_data["error"] = {  # mutable-ok: VideoObject expects a mutable error payload.
                "code": "generation_failed",
                "message": str(response_data.get("status")),
            }
        if response_data.get("duration") is not None:
            video_data["seconds"] = str(response_data["duration"])
        if response_data.get("resolution") is not None:
            video_data["size"] = str(response_data["resolution"])

        video_obj = VideoObject(**video_data)
        self._wrap_video_id(video_obj, custom_llm_provider, model)
        return video_obj

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: BaseVideoConfig passes a mutable header mapping.
        extra_body: dict | None = None,  # mutable-ok: BaseVideoConfig defines provider extras as a dict.
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig requires a dict payload.
        raise NotImplementedError("Video remix is not supported by the MiniMax v1 API")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported by the MiniMax v1 API")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: BaseVideoConfig passes a mutable header mapping.
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: dict | None = None,  # mutable-ok: BaseVideoConfig defines query extras as a dict.
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig requires a dict payload.
        raise NotImplementedError("Video listing is not supported by the MiniMax v1 API")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseVideoConfig requires a dict response.
        raise NotImplementedError("Video listing is not supported by the MiniMax v1 API")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: BaseVideoConfig passes a mutable header mapping.
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig requires a dict payload.
        raise NotImplementedError("Video deletion is not supported by the MiniMax v1 API")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        raise NotImplementedError("Video deletion is not supported by the MiniMax v1 API")

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | httpx.Headers,  # mutable-ok: BaseLLMException accepts the response header mapping.
    ) -> BaseLLMException:
        return BaseLLMException(status_code=status_code, message=error_message, headers=headers)

    @staticmethod
    def _coerce_duration(value: object) -> object:
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return value

    @staticmethod
    def _map_status(status: object) -> str:
        normalized = str(status or "").strip().lower().replace(" ", "_")
        if normalized in ("success", "succeeded", "completed", "complete"):
            return "completed"
        if normalized in ("fail", "failed", "error", "cancelled", "canceled"):
            return "failed"
        if normalized in ("queueing", "queued", "preparing", "pending"):
            return "queued"
        return "in_progress"

    @staticmethod
    def _add_request_metadata(
        video_data: dict,  # mutable-ok: VideoObject metadata is assembled in place.
        request_data: dict | None,  # mutable-ok: Request metadata arrives as a dict.
    ) -> None:
        if not request_data:
            return
        if request_data.get("duration") is not None:
            video_data["seconds"] = str(request_data["duration"])
        if request_data.get("resolution") is not None:
            video_data["size"] = str(request_data["resolution"])

    @staticmethod
    def _usage_from_video(video_obj: VideoObject) -> dict:  # mutable-ok: VideoObject usage requires a dict.
        if video_obj.seconds is None:
            return {}  # mutable-ok: VideoObject usage requires a dict.
        try:
            return {  # mutable-ok: VideoObject usage requires a dict.
                "duration_seconds": float(video_obj.seconds)
            }
        except (TypeError, ValueError):
            return {}  # mutable-ok: VideoObject usage requires a dict.

    @staticmethod
    def _wrap_video_id(video_obj: VideoObject, provider: str | None, model: str | None) -> None:
        if provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(video_obj.id, provider, model)

    def _parse_json_response(
        self, raw_response: httpx.Response
    ) -> dict:  # mutable-ok: JSON objects are provider dicts.
        self._raise_for_status(raw_response)
        try:
            return raw_response.json()
        except Exception as exc:
            raise ValueError(f"MiniMax returned an invalid JSON response: {exc}") from exc

    def _raise_for_provider_error(
        self,
        raw_response: httpx.Response,
        response_data: dict,  # mutable-ok: Parsed provider JSON is represented as a dict.
    ) -> None:
        self._raise_for_status(raw_response)
        base_resp = response_data.get("base_resp") or {}  # mutable-ok: Missing provider metadata uses an empty dict.
        status_code = base_resp.get("status_code")
        if status_code not in (None, 0, "0"):
            message = base_resp.get("status_msg") or "MiniMax video request failed"
            raise self.get_error_class(str(message), raw_response.status_code, raw_response.headers)

    def _raise_for_status(self, raw_response: httpx.Response) -> None:
        if raw_response.status_code >= 400:
            raise self.get_error_class(raw_response.text, raw_response.status_code, raw_response.headers)

    @staticmethod
    def _request_headers(
        raw_response: httpx.Response,
    ) -> dict[str, str]:  # mutable-ok: HTTP handlers require a dict of headers.
        request = getattr(raw_response, "_request", None)
        request_headers = getattr(request, "headers", None)
        if isinstance(request_headers, (dict, httpx.Headers)):
            authorization = request_headers.get("Authorization")
            if authorization:
                return {"Authorization": authorization}  # mutable-ok: HTTP handlers require a dict of headers.
        return {}  # mutable-ok: HTTP handlers require a dict of headers.

    @staticmethod
    def _api_base_from_response(raw_response: httpx.Response) -> str:
        request = getattr(raw_response, "_request", None)
        request_url = getattr(request, "url", None)
        if request_url is None:
            return "https://api.minimax.io/v1"
        parsed = urlsplit(str(request_url))
        path = parsed.path
        v1_index = path.find("/v1/")
        root_path = path[: v1_index + len("/v1")] if v1_index >= 0 else "/v1"
        return urlunsplit((parsed.scheme, parsed.netloc, root_path, "", ""))

    @staticmethod
    def _is_binary_response(raw_response: httpx.Response) -> bool:
        content_type = raw_response.headers.get("content-type", "")
        return content_type.startswith("video/") or content_type == "application/octet-stream"

    @staticmethod
    def _get_download_url(
        response_data: dict,  # mutable-ok: Parsed provider JSON is represented as a dict.
    ) -> str | None:
        file_data = response_data.get("file")
        if isinstance(file_data, dict):
            for key in ("download_url", "url"):
                if file_data.get(key):
                    return str(file_data[key])
        for key in ("download_url", "url"):
            if response_data.get(key):
                return str(response_data[key])
        return None

    @staticmethod
    def _prepare_first_frame_image(image: object) -> object:
        if isinstance(image, str):
            return image

        content = image
        content_type = None
        if isinstance(image, tuple):
            if len(image) < 2:
                raise ValueError("MiniMax input_reference tuple must include file content")
            content = image[1]
            if len(image) >= 3 and isinstance(image[2], str):
                content_type = image[2]

        if isinstance(content, PathLike):
            with open(content, "rb") as image_file:
                image_bytes = image_file.read()
        elif isinstance(content, bytes):
            image_bytes = content
        elif hasattr(content, "read"):
            current_position = content.tell() if hasattr(content, "tell") else None
            if hasattr(content, "seek"):
                content.seek(0)
            image_bytes = content.read()
            if current_position is not None and hasattr(content, "seek"):
                content.seek(current_position)
        else:
            raise TypeError("MiniMax input_reference must be a URL, path, bytes, or file object")

        if not isinstance(image_bytes, bytes):
            raise TypeError("MiniMax input_reference file content must be bytes")
        content_type = content_type or ImageEditRequestUtils.get_image_content_type(image_bytes)
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{content_type};base64,{encoded}"
