"""Video generation for Hosted VLLM (vLLM-Omni OpenAI-compatible /v1/videos)."""

import json
from collections.abc import Mapping
from io import BufferedReader
from types import MappingProxyType
from typing import Final
from urllib.parse import urlparse

from httpx._types import FileTypes, RequestFiles

from litellm.images.utils import ImageEditRequestUtils
from litellm.litellm_core_utils.url_utils import SSRFError, validate_url
from litellm.llms.openai.videos.transformation import OpenAIVideoConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams

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


def _reject_unsafe_media_url(url: str) -> None:
    scheme: Final = urlparse(url).scheme.lower()
    if scheme in ("", "data"):
        return
    if scheme not in ("http", "https"):
        raise SSRFError(f"URL scheme '{scheme}' is not allowed")
    validate_url(url)


def _reject_unsafe_urls_in_item(url_key: str, item: object) -> None:
    if not isinstance(item, Mapping):
        return
    url: Final = item.get(url_key)
    if isinstance(url, str):
        _reject_unsafe_media_url(url)


def _reject_unsafe_media_urls(field_name: str, value: object) -> None:
    url_key: Final = _REFERENCE_URL_KEYS.get(field_name)
    if url_key is None:
        return
    parsed: Final = _maybe_json(value)
    if isinstance(parsed, list):
        for item in parsed:
            _reject_unsafe_urls_in_item(url_key, item)
        return
    if isinstance(parsed, Mapping):
        _reject_unsafe_urls_in_item(url_key, parsed)


def _form_value(key: str, value: object) -> str:
    _reject_unsafe_media_urls(key, value)
    return _serialize_form_value(value)


def _input_reference_file(reference: object) -> tuple[str, FileTypes]:
    content_type: Final = ImageEditRequestUtils.get_image_content_type(reference)
    if isinstance(reference, BufferedReader):
        return ("input_reference", (reference.name, reference, content_type))
    return ("input_reference", ("input_reference.png", reference, content_type))


class HostedVLLMVideoConfig(OpenAIVideoConfig):
    """
    vLLM-Omni videos API is OpenAI-compatible but requires multipart/form-data.

    https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/videos_api/
    """

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
        data: Final = {  # mutable-ok: BaseVideoConfig contract returns a data dict
            "model": model,
            "prompt": prompt,
            **{  # mutable-ok: spread remaining Omni form fields into that data dict
                key: _form_value(key, value)
                for key, value in video_create_optional_request_params.items()
                if key not in _EXCLUDED_FORM_KEYS and value is not None
            },
        }
        input_reference: Final = video_create_optional_request_params.get("input_reference")
        if input_reference is None:
            return data, (), api_base
        return data, (_input_reference_file(input_reference),), api_base
