"""Video generation for Hosted VLLM (vLLM-Omni OpenAI-compatible /v1/videos)."""

import base64
import json
import mimetypes
from collections.abc import Mapping
from dataclasses import dataclass
from io import BufferedReader
from types import MappingProxyType
from typing import Final
from urllib.parse import urlparse

from httpx import Response
from httpx._types import FileTypes, RequestFiles

from litellm.constants import (
    MAX_IMAGE_URL_DOWNLOAD_SIZE_MB,
    MAX_VIDEO_MEDIA_URL_TOTAL_DOWNLOAD_SIZE_MB,
    MAX_VIDEO_MEDIA_URLS_PER_REQUEST,
)
from litellm.images.utils import ImageEditRequestUtils
from litellm.litellm_core_utils.url_utils import SSRFError, safe_get
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.llms.openai.videos.transformation import OpenAIVideoConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams

_BYTES_PER_MB: Final = 1024 * 1024
_STREAM_CHUNK_SIZE: Final = 8192

_EXCLUDED_FORM_KEYS: Final = frozenset(
    {
        "model",
        "prompt",
        "extra_headers",
        "extra_query",
        "extra_body",
        "timeout",
        "custom_llm_provider",
        "input_reference",
        "characters",
    }
)

_VLLM_OMNI_VIDEO_PARAMS: Final = (
    "image_reference",
    "video_reference",
    "audio_reference",
    "width",
    "height",
    "num_frames",
    "fps",
    "num_inference_steps",
    "guidance_scale",
    "guidance_scale_2",
    "boundary_ratio",
    "flow_shift",
    "true_cfg_scale",
    "seed",
    "generate_sound",
    "sound_duration",
    "negative_prompt",
    "enable_frame_interpolation",
    "frame_interpolation_exp",
    "frame_interpolation_scale",
    "frame_interpolation_model_path",
    "lora",
    "extra_params",
    "aspect_ratio",
)

_REFERENCE_URL_KEYS: Final = MappingProxyType(
    {
        "image_reference": "image_url",
        "video_reference": "video_url",
        "audio_reference": "audio_url",
    }
)


@dataclass(slots=True)
class _MediaDownloadBudget:
    remaining_urls: int
    remaining_bytes: int
    max_bytes_per_url: int

    def consume_url_slot(self) -> None:
        if self.max_bytes_per_url <= 0:
            raise ValueError("remote media URL download is disabled (MAX_IMAGE_URL_DOWNLOAD_SIZE_MB=0)")
        if self.remaining_urls < 1:
            raise ValueError("too many remote media URL references on one video request")
        self.remaining_urls -= 1

    def max_read_bytes(self) -> int:
        return min(self.max_bytes_per_url, self.remaining_bytes)

    def consume_bytes(self, nbytes: int) -> None:
        if nbytes > self.remaining_bytes:
            raise ValueError("remote media download exceeded the per-request size limit")
        self.remaining_bytes -= nbytes


def _serialize_form_value(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value)
    return str(value)


def _maybe_json(value: object) -> object:
    if not isinstance(value, str):
        return value
    stripped: Final = value.strip()
    if not stripped or stripped[0] not in "{[":
        return value
    return json.loads(stripped)


def _content_type_from_headers(headers: Mapping[str, object]) -> str:
    raw: Final = headers.get("content-type", "application/octet-stream")
    if not isinstance(raw, str) or not raw.strip():
        return "application/octet-stream"
    return raw.split(";", 1)[0].strip() or "application/octet-stream"


def _declared_content_length(headers: Mapping[str, object]) -> int | None:
    raw: Final = headers.get("content-length")
    if not isinstance(raw, str):
        return None
    stripped: Final = raw.strip()
    if not stripped.isdigit():
        return None
    return int(stripped)


def _read_capped_body(response: Response, max_bytes: int) -> bytes:
    declared: Final = _declared_content_length(response.headers)
    if declared is not None and declared > max_bytes:
        response.close()
        raise ValueError("remote media URL Content-Length exceeds the maximum allowed size")
    body: Final = bytearray()  # mutable-ok: streaming accumulator with a hard cap
    for chunk in response.iter_bytes(chunk_size=_STREAM_CHUNK_SIZE):
        body.extend(chunk)
        if len(body) > max_bytes:
            response.close()
            raise ValueError("remote media download exceeded the maximum allowed size")
    return bytes(body)


def _fetch_url_as_data_url(url: str, client: HTTPHandler, budget: _MediaDownloadBudget) -> str:
    budget.consume_url_slot()
    response: Final = safe_get(client, url)
    response.raise_for_status()
    content: Final = _read_capped_body(response, budget.max_read_bytes())
    budget.consume_bytes(len(content))
    encoded: Final = base64.b64encode(content).decode("ascii")
    return f"data:{_content_type_from_headers(response.headers)};base64,{encoded}"


def _inline_reference_item(url_key: str, item: object, client: HTTPHandler, budget: _MediaDownloadBudget) -> object:
    if not isinstance(item, Mapping):
        return item
    url: Final = item.get(url_key)
    if not isinstance(url, str):
        return item
    scheme: Final = urlparse(url).scheme.lower()
    if scheme in ("", "data"):
        return item
    if scheme not in ("http", "https"):
        raise SSRFError(f"URL scheme '{scheme}' is not allowed")
    return {  # mutable-ok: JSON form field serialized immediately
        **item,
        url_key: _fetch_url_as_data_url(url, client, budget),
    }


def _inline_media_reference(
    field_name: str, value: object, client: HTTPHandler, budget: _MediaDownloadBudget
) -> object:
    url_key: Final = _REFERENCE_URL_KEYS.get(field_name)
    if url_key is None:
        return value
    parsed: Final = _maybe_json(value)
    if isinstance(parsed, list):
        return tuple(_inline_reference_item(url_key, item, client, budget) for item in parsed)
    if isinstance(parsed, Mapping):
        return _inline_reference_item(url_key, parsed, client, budget)
    return value


def _input_reference_file(reference: object) -> tuple[str, FileTypes]:
    if isinstance(reference, BufferedReader):
        reader_name: Final = reference.name
        reader_type: Final = mimetypes.guess_type(reader_name)[0] or ImageEditRequestUtils.get_image_content_type(
            reference
        )
        return ("input_reference", (reader_name, reference, reader_type))

    fallback_name: Final = getattr(reference, "name", None) or "input_reference.png"
    fallback_type: Final = mimetypes.guess_type(str(fallback_name))[0] or ImageEditRequestUtils.get_image_content_type(
        reference
    )
    return ("input_reference", (str(fallback_name), reference, fallback_type))


class HostedVLLMVideoConfig(OpenAIVideoConfig):
    """
    vLLM-Omni videos API is OpenAI-compatible but requires multipart/form-data.

    https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/videos_api/
    """

    def __init__(
        self,
        media_http_client: HTTPHandler | None = None,
        *,
        max_media_bytes_per_url: int | None = None,
        max_media_bytes_per_request: int | None = None,
        max_media_urls_per_request: int | None = None,
    ) -> None:
        super().__init__()
        self._media_http_client = media_http_client
        self._max_media_bytes_per_url = max_media_bytes_per_url
        self._max_media_bytes_per_request = max_media_bytes_per_request
        self._max_media_urls_per_request = max_media_urls_per_request

    def _media_budget(self) -> _MediaDownloadBudget:
        per_url: Final = (
            self._max_media_bytes_per_url
            if self._max_media_bytes_per_url is not None
            else int(MAX_IMAGE_URL_DOWNLOAD_SIZE_MB * _BYTES_PER_MB)
        )
        per_request: Final = (
            self._max_media_bytes_per_request
            if self._max_media_bytes_per_request is not None
            else int(MAX_VIDEO_MEDIA_URL_TOTAL_DOWNLOAD_SIZE_MB * _BYTES_PER_MB)
        )
        url_slots: Final = (
            self._max_media_urls_per_request
            if self._max_media_urls_per_request is not None
            else MAX_VIDEO_MEDIA_URLS_PER_REQUEST
        )
        return _MediaDownloadBudget(
            remaining_urls=url_slots,
            remaining_bytes=per_request,
            max_bytes_per_url=per_url,
        )

    def get_supported_openai_params(self, model: str) -> list:  # mutable-ok: BaseVideoConfig contract
        return [  # mutable-ok: BaseVideoConfig returns list
            *super().get_supported_openai_params(model),
            *_VLLM_OMNI_VIDEO_PARAMS,
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: BaseVideoConfig contract; extra_body merge mutates this dict
        return {  # mutable-ok: VideoGenerationRequestUtils.update/pop extra_body onto this mapping
            key: value for key, value in video_create_optional_params.items() if value is not None
        }

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: BaseVideoConfig contract
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict:  # mutable-ok: BaseVideoConfig contract
        resolved_key: Final = (
            (litellm_params.api_key if litellm_params is not None else None)
            or api_key
            or get_secret_str("HOSTED_VLLM_API_KEY")
            or "fake-api-key"
        )
        return {**headers, "Authorization": f"Bearer {resolved_key}"}  # mutable-ok: httpx headers are a dict

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,  # mutable-ok: BaseVideoConfig contract
    ) -> str:
        resolved_api_base: Final = api_base or get_secret_str("HOSTED_VLLM_API_BASE")
        if resolved_api_base is None:
            raise ValueError(
                "api_base not set for Hosted VLLM videos API. "
                "Set via api_base parameter or HOSTED_VLLM_API_BASE environment variable"
            )
        trimmed: Final = resolved_api_base.rstrip("/")
        if trimmed.endswith("/v1"):
            return f"{trimmed}/videos"
        return f"{trimmed}/v1/videos"

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: dict,  # mutable-ok: BaseVideoConfig contract
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: BaseVideoConfig contract
    ) -> tuple[dict, RequestFiles, str]:  # mutable-ok: BaseVideoConfig contract
        media_client: Final = self._media_http_client or HTTPHandler(concurrent_limit=1)
        media_budget: Final = self._media_budget()
        input_reference: Final = video_create_optional_request_params.get("input_reference")
        form_files: Final = tuple(
            (
                key,
                (None, _serialize_form_value(_inline_media_reference(key, value, media_client, media_budget))),
            )
            for key, value in video_create_optional_request_params.items()
            if key not in _EXCLUDED_FORM_KEYS and value is not None
        )
        reference_files: Final = (_input_reference_file(input_reference),) if input_reference is not None else ()
        files: Final = (
            ("model", (None, model)),
            ("prompt", (None, prompt)),
            *form_files,
            *reference_files,
        )
        return {}, files, api_base  # mutable-ok: empty data dict; files carry the multipart fields
