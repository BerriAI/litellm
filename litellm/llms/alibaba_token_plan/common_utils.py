import base64
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

import litellm
from litellm.images.utils import ImageEditRequestUtils
from litellm.secret_managers.main import get_secret_str

DEFAULT_API_BASE: Final = "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
MESSAGES_PATH: Final = "apps/anthropic"
IMAGE_PATH: Final = "api/v1/services/aigc/multimodal-generation/generation"
SPEECH_PATH: Final = "api/v1/services/audio/tts/SpeechSynthesizer"
REALTIME_PATH: Final = "api-ws/v1/realtime"
VIDEO_PATH: Final = "api/v1/services/aigc/video-generation/video-synthesis"


@runtime_checkable
class _Readable(Protocol):
    def read(self) -> bytes: ...


def get_api_key(api_key: str | None) -> str | None:
    return api_key or get_secret_str("ALIBABA_TOKEN_PLAN_API_KEY") or litellm.api_key


def require_api_key(api_key: str | None) -> str:
    resolved_key: Final = get_api_key(api_key)
    if not resolved_key:
        raise litellm.AuthenticationError(
            message="ALIBABA_TOKEN_PLAN_API_KEY is not set; provide a Token Plan api_key",
            llm_provider="alibaba_token_plan",
            model="",
        )
    return resolved_key


def get_api_base(api_base: str | None) -> str:
    return (api_base or get_secret_str("ALIBABA_TOKEN_PLAN_API_BASE") or DEFAULT_API_BASE).rstrip("/")


def get_api_url(api_base: str | None, path: str) -> str:
    return f"{get_api_base(api_base).removesuffix('/compatible-mode/v1')}/{path}"


def validate_headers(
    headers: Mapping[str, str], api_key: str | None
) -> dict[str, str]:  # mutable-ok: provider interface requires a mutable return value
    return {**headers, "Authorization": f"Bearer {require_api_key(api_key)}", "Content-Type": "application/json"}


def image_reference(image: object) -> str:
    source: Final = image[1] if isinstance(image, tuple) and len(image) > 1 else image
    if isinstance(source, str):
        return source
    data: Final = (
        Path(source).read_bytes()
        if isinstance(source, PathLike)
        else source.read()
        if isinstance(source, _Readable)
        else source
    )
    if not isinstance(data, bytes):
        raise TypeError("Image must be a URL, data URI, bytes, path or binary file")
    content_type: Final = ImageEditRequestUtils.get_image_content_type(data)
    return f"data:{content_type};base64,{base64.b64encode(data).decode('ascii')}"
