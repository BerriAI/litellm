import base64
import mimetypes
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast
from urllib.parse import urlparse

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    encode_video_id_with_provider,
    extract_original_video_id,
)

from ..common_utils import (
    VolcEngineError,
    get_volcengine_base_url,
    get_volcengine_headers,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class VolcEngineVideoConfig(BaseVideoConfig):
    """
    Volcengine (Ark) video generation task API.

    Official task endpoints:
    - POST   /api/v3/contents/generations/tasks
    - GET    /api/v3/contents/generations/tasks/{id}
    - GET    /api/v3/contents/generations/tasks
    - DELETE /api/v3/contents/generations/tasks/{id}

    Download is performed by first fetching task status and then downloading
    the `content.video_url` returned by the task result.
    """

    _DEFAULT_DOWNLOAD_MIME_TYPE = "video/mp4"

    def __init__(self) -> None:
        super().__init__()

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "model",
            "prompt",
            "input_reference",
            "seconds",
            "size",
            "content",
            "duration",
            "ratio",
            "generate_audio",
            "watermark",
            "resolution",
            "seed",
            "framespersecond",
            "fps",
            "extra_headers",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        mapped_params: Dict[str, Any] = {}

        for key, value in dict(video_create_optional_params).items():
            if key in {"model", "prompt", "extra_headers"}:
                continue
            if key == "size":
                aspect_ratio = self._convert_size_to_ratio(cast(Optional[str], value))
                if aspect_ratio is not None:
                    mapped_params["ratio"] = aspect_ratio
            elif key == "seconds":
                duration = self._coerce_duration(value)
                if duration is not None:
                    mapped_params["duration"] = duration
            else:
                mapped_params[key] = value

        return mapped_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[GenericLiteLLMParams] = None,
    ) -> dict:
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key

        api_key = (
            api_key
            or litellm.api_key
            or get_secret_str("ARK_API_KEY")
            or get_secret_str("VOLCENGINE_API_KEY")
        )

        if api_key is None:
            raise ValueError(
                "Volcengine API key is required. Set ARK_API_KEY / VOLCENGINE_API_KEY or pass api_key."
            )

        return get_volcengine_headers(api_key=api_key, extra_headers=headers)

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        base_url = (
            api_base
            or litellm.api_base
            or get_secret_str("VOLCENGINE_API_BASE")
            or get_secret_str("ARK_API_BASE")
            or get_volcengine_base_url()
        )
        base_url = base_url.rstrip("/")

        if base_url.endswith("/contents/generations/tasks"):
            return base_url
        if base_url.endswith("/api/v3"):
            return f"{base_url}/contents/generations/tasks"
        return f"{base_url}/api/v3/contents/generations/tasks"

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles, str]:
        request_params = {
            k: deepcopy(v)
            for k, v in video_create_optional_request_params.items()
            if k not in {"model", "extra_headers", "prompt"}
        }

        content = self._build_content_list(
            prompt=prompt,
            provided_content=request_params.pop("content", None),
            input_reference=request_params.pop("input_reference", None),
        )

        request_data: Dict[str, Any] = {
            "model": self._normalize_model(model),
            "content": content,
        }
        request_data.update(request_params)

        files_list: List[Tuple[str, Any]] = []
        return request_data, files_list, api_base

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
        request_data: Optional[Dict] = None,
    ) -> VideoObject:
        response_data = raw_response.json()
        video_obj = self._task_to_video_object(
            task=response_data,
            custom_llm_provider=custom_llm_provider,
            model_id=model,
            fallback_model=self._normalize_model(model),
        )

        if request_data is not None:
            if video_obj.model is None:
                video_obj.model = request_data.get("model") or self._normalize_model(model)
            if video_obj.seconds is None and request_data.get("duration") is not None:
                video_obj.seconds = str(request_data["duration"])
            if video_obj.size is None and request_data.get("ratio") is not None:
                video_obj.size = str(request_data["ratio"])

            usage = dict(video_obj.usage or {})
            duration = request_data.get("duration")
            if duration is not None:
                try:
                    usage["duration_seconds"] = float(duration)
                except (TypeError, ValueError):
                    pass
            if usage:
                video_obj.usage = usage

            hidden_params = dict(getattr(video_obj, "_hidden_params", {}) or {})
            hidden_params["request_content"] = request_data.get("content")
            hidden_params["response_cost"] = 0.0
            video_obj._hidden_params = hidden_params

        return video_obj

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        variant: Optional[str] = None,
    ) -> Tuple[str, Dict]:
        original_video_id = extract_original_video_id(video_id)
        url = f"{api_base.rstrip('/')}/{original_video_id}"
        return url, {}

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        response_data = raw_response.json()
        video_url = self._extract_video_download_url(response_data)

        httpx_client: HTTPHandler = _get_httpx_client()
        # Preserve the signed query string as returned by Volcengine. Some download
        # URLs include UTF-8 filename metadata in the query, and rebuilding params
        # via HTTPHandler.get() can break those URLs during re-encoding.
        video_response = httpx_client.client.get(video_url)
        video_response.raise_for_status()
        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        response_data = raw_response.json()
        video_url = self._extract_video_download_url(response_data)

        async_httpx_client: AsyncHTTPHandler = get_async_httpx_client(
            llm_provider=LlmProviders.VOLCENGINE
        )
        # Preserve the original signed query string for the same reason as the
        # sync download path above.
        video_response = await async_httpx_client.client.get(video_url)
        video_response.raise_for_status()
        return video_response.content

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("video remix is not supported for Volcengine")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        raise NotImplementedError("video remix is not supported for Volcengine")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        extra_query: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        params: Dict[str, Any] = {}

        if limit is not None:
            params["page_size"] = str(limit)

        if after is not None:
            after_str = str(after)
            if after_str.isdigit():
                params["page_num"] = after_str
            else:
                params["filter.task_ids"] = extract_original_video_id(after_str)

        if extra_query:
            params.update(extra_query)

        return api_base, params

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        response_data = raw_response.json()
        tasks = response_data.get("items", []) or []

        transformed_items: List[Dict[str, Any]] = []
        for task in tasks:
            video_obj = self._task_to_video_object(
                task=task,
                custom_llm_provider=custom_llm_provider,
                model_id=task.get("model"),
                fallback_model=task.get("model"),
            )
            item_dict = video_obj.model_dump(exclude_none=True)
            if isinstance(task.get("content"), dict):
                item_dict["content"] = task["content"]
            transformed_items.append(item_dict)

        transformed_response = dict(response_data)
        transformed_response["object"] = "list"
        transformed_response["data"] = transformed_items

        if transformed_items:
            transformed_response["first_id"] = transformed_items[0].get("id")
            transformed_response["last_id"] = transformed_items[-1].get("id")

        total = response_data.get("total")
        page_num = self._safe_int(response_data.get("page_num"))
        page_size = self._safe_int(response_data.get("page_size"))
        if total is not None and page_num is not None and page_size:
            transformed_response["has_more"] = page_num * page_size < int(total)

        return transformed_response

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        original_video_id = extract_original_video_id(video_id)
        url = f"{api_base.rstrip('/')}/{original_video_id}"
        return url, {}

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        try:
            response_data = raw_response.json()
        except Exception:
            response_data = {}

        task_id = response_data.get("id") or self._extract_task_id_from_request(raw_response)
        status = self._map_status(response_data.get("status") or "deleted")

        video_obj = VideoObject(
            id=task_id or "",
            object="video",
            status=status,
        )
        video_obj._hidden_params = {"raw_response": response_data}
        return video_obj

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        original_video_id = extract_original_video_id(video_id)
        url = f"{api_base.rstrip('/')}/{original_video_id}"
        return url, {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        response_data = raw_response.json()
        return self._task_to_video_object(
            task=response_data,
            custom_llm_provider=custom_llm_provider,
            model_id=response_data.get("model"),
            fallback_model=response_data.get("model"),
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> VolcEngineError:
        typed_headers = (
            headers if isinstance(headers, httpx.Headers) else httpx.Headers(headers or {})
        )
        return VolcEngineError(
            status_code=status_code,
            message=error_message,
            headers=typed_headers,
        )

    def transform_video_create_character_request(
        self, name, video, api_base, litellm_params, headers
    ):
        raise NotImplementedError("video create character is not supported for Volcengine")

    def transform_video_create_character_response(self, raw_response, logging_obj):
        raise NotImplementedError("video create character is not supported for Volcengine")

    def transform_video_get_character_request(
        self, character_id, api_base, litellm_params, headers
    ):
        raise NotImplementedError("video get character is not supported for Volcengine")

    def transform_video_get_character_response(self, raw_response, logging_obj):
        raise NotImplementedError("video get character is not supported for Volcengine")

    def transform_video_edit_request(
        self, prompt, video_id, api_base, litellm_params, headers, extra_body=None
    ):
        raise NotImplementedError("video edit is not supported for Volcengine")

    def transform_video_edit_response(
        self, raw_response, logging_obj, custom_llm_provider=None
    ):
        raise NotImplementedError("video edit is not supported for Volcengine")

    def transform_video_extension_request(
        self,
        prompt,
        video_id,
        seconds,
        api_base,
        litellm_params,
        headers,
        extra_body=None,
    ):
        raise NotImplementedError("video extension is not supported for Volcengine")

    def transform_video_extension_response(
        self, raw_response, logging_obj, custom_llm_provider=None
    ):
        raise NotImplementedError("video extension is not supported for Volcengine")

    def _build_content_list(
        self,
        prompt: str,
        provided_content: Any,
        input_reference: Any,
    ) -> List[Dict[str, Any]]:
        if provided_content is None:
            content: List[Dict[str, Any]] = []
        elif isinstance(provided_content, list):
            content = [deepcopy(item) for item in provided_content]
        else:
            raise ValueError("Volcengine `content` must be a list of content blocks.")

        if prompt and not any(
            isinstance(item, dict) and item.get("type") == "text" for item in content
        ):
            content.insert(0, {"type": "text", "text": prompt})

        if input_reference is not None:
            content.append(self._convert_reference_to_content_item(input_reference))

        if not content:
            raise ValueError(
                "Volcengine video generation requires either `prompt` or `content`."
            )

        return content

    def _normalize_model(self, model: str) -> str:
        if model.startswith("volcengine/"):
            return model.split("/", 1)[1]
        return model

    def _coerce_duration(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _convert_size_to_ratio(self, size: Optional[str]) -> Optional[str]:
        if not size:
            return None
        if ":" in size:
            return size
        if "x" not in size:
            return size

        width_str, height_str = size.lower().split("x", 1)
        try:
            width = int(width_str)
            height = int(height_str)
        except ValueError:
            return None

        from math import gcd

        divisor = gcd(width, height)
        if divisor == 0:
            return None
        return f"{width // divisor}:{height // divisor}"

    def _convert_reference_to_content_item(self, reference: Any) -> Dict[str, Any]:
        if isinstance(reference, dict):
            if reference.get("type") in {"image_url", "video_url", "audio_url"}:
                return deepcopy(reference)
            if "url" in reference:
                media_type, role = self._infer_media_type_and_role(str(reference["url"]))
                return {
                    "type": media_type,
                    media_type: {"url": reference["url"]},
                    "role": role,
                }
            raise ValueError("Unsupported Volcengine reference dict format.")

        if isinstance(reference, str):
            media_type, role = self._infer_media_type_and_role(reference)
            return {
                "type": media_type,
                media_type: {"url": reference},
                "role": role,
            }

        if isinstance(reference, bytes):
            data_uri = self._bytes_to_data_uri(reference, self._DEFAULT_DOWNLOAD_MIME_TYPE)
            return {
                "type": "video_url",
                "video_url": {"url": data_uri},
                "role": "reference_video",
            }

        if hasattr(reference, "read"):
            file_bytes = reference.read()
            if hasattr(reference, "seek"):
                reference.seek(0)
            mime_type = (
                mimetypes.guess_type(getattr(reference, "name", ""))[0]
                or "image/png"
            )
            media_type, role = self._infer_media_type_and_role(
                getattr(reference, "name", ""), mime_type=mime_type
            )
            data_uri = self._bytes_to_data_uri(file_bytes, mime_type)
            return {
                "type": media_type,
                media_type: {"url": data_uri},
                "role": role,
            }

        raise ValueError(
            "Volcengine `input_reference` must be a URL string, bytes, or file-like object."
        )

    def _infer_media_type_and_role(
        self, value: str, mime_type: Optional[str] = None
    ) -> Tuple[str, str]:
        inferred_mime = mime_type or mimetypes.guess_type(urlparse(value).path)[0] or ""
        if inferred_mime.startswith("video/"):
            return "video_url", "reference_video"
        if inferred_mime.startswith("audio/"):
            return "audio_url", "reference_audio"
        return "image_url", "reference_image"

    def _bytes_to_data_uri(self, value: bytes, mime_type: str) -> str:
        encoded = base64.b64encode(value).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def _task_to_video_object(
        self,
        task: Dict[str, Any],
        custom_llm_provider: Optional[str],
        model_id: Optional[str],
        fallback_model: Optional[str] = None,
    ) -> VideoObject:
        task_id = task.get("id", "")
        response_model = task.get("model") or fallback_model
        status = self._map_status(task.get("status") or "queued")
        created_at = self._safe_int(task.get("created_at"))
        updated_at = self._safe_int(task.get("updated_at"))
        duration = task.get("duration")

        video_id = task_id
        if custom_llm_provider and task_id:
            video_id = encode_video_id_with_provider(
                task_id,
                custom_llm_provider,
                model_id or response_model,
            )

        video_obj = VideoObject(
            id=video_id,
            object="video",
            status=status,
            created_at=created_at,
            completed_at=self._infer_completed_at(task.get("status"), updated_at),
            error=self._normalize_error(task.get("error")),
            progress=self._infer_progress(task),
            seconds=str(duration) if duration is not None else None,
            size=task.get("ratio"),
            model=response_model,
        )

        usage = {}
        if isinstance(task.get("usage"), dict):
            usage.update(task["usage"])
        if duration is not None:
            try:
                usage["duration_seconds"] = float(duration)
            except (TypeError, ValueError):
                pass
        if usage:
            video_obj.usage = usage

        hidden_params = {
            "raw_response": task,
            "raw_status": task.get("status"),
            "response_cost": 0.0,
            "content": task.get("content"),
            "video_url": self._get_nested(task, "content", "video_url"),
            "audio_url": self._get_nested(task, "content", "audio_url"),
            "ratio": task.get("ratio"),
            "resolution": task.get("resolution"),
            "framespersecond": task.get("framespersecond"),
            "service_tier": task.get("service_tier"),
            "execution_expires_after": task.get("execution_expires_after"),
            "generate_audio": task.get("generate_audio"),
            "draft": task.get("draft"),
            "draft_task_id": task.get("draft_task_id"),
            "updated_at": updated_at,
        }
        video_obj._hidden_params = {
            key: value for key, value in hidden_params.items() if value is not None
        }
        return video_obj

    def _extract_video_download_url(self, response_data: Dict[str, Any]) -> str:
        video_url = self._get_nested(response_data, "content", "video_url")
        if isinstance(video_url, str) and video_url:
            return video_url

        raw_status = str(response_data.get("status", "")).lower()
        if raw_status in {"queued", "running"}:
            raise ValueError(
                f"Video is still processing (status: {raw_status}). Please check status before downloading."
            )
        if raw_status in {"failed", "expired", "cancelled"}:
            raise ValueError(
                f"Video generation is not downloadable because task status is {raw_status}."
            )
        raise ValueError("Volcengine response did not include `content.video_url`.")

    def _extract_task_id_from_request(self, raw_response: httpx.Response) -> Optional[str]:
        request = getattr(raw_response, "request", None)
        request_url = getattr(request, "url", None)
        if request_url is None:
            return None
        return Path(request_url.path).name or None

    def _normalize_error(self, error: Any) -> Optional[Dict[str, Any]]:
        if error in (None, {}):
            return None
        if isinstance(error, dict):
            return error
        return {"message": str(error)}

    def _map_status(self, provider_status: str) -> str:
        status_map = {
            "queued": "queued",
            "running": "processing",
            "cancelled": "cancelled",
            "succeeded": "completed",
            "failed": "failed",
            "expired": "expired",
            "deleted": "deleted",
        }
        return status_map.get(str(provider_status).lower(), str(provider_status).lower())

    def _infer_completed_at(
        self, provider_status: Optional[str], updated_at: Optional[int]
    ) -> Optional[int]:
        if updated_at is None:
            return None
        if str(provider_status).lower() in {"succeeded", "failed", "cancelled", "expired"}:
            return updated_at
        return None

    def _infer_progress(self, task: Dict[str, Any]) -> Optional[int]:
        progress = self._safe_int(task.get("progress"))
        if progress is not None:
            return progress

        raw_status = str(task.get("status", "")).lower()
        if raw_status == "queued":
            return 0
        if raw_status == "running":
            return 50
        if raw_status == "succeeded":
            return 100
        return None

    def _safe_int(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _get_nested(self, data: Dict[str, Any], *keys: str) -> Any:
        current: Any = data
        for key in keys:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current
