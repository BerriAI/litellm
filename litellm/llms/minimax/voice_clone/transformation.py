"""MiniMax voice-cloning API transformations.

MiniMax voice cloning is a two-step operation:

1. Upload an audio sample to ``/v1/files/upload`` with a voice-cloning
   purpose and read the returned ``file_id``.
2. Submit that ``file_id`` with a caller-provided ``voice_id`` and supported
   speech model to ``/v1/voice_clone``.

The API is available from both the international and China hosts.  This
module deliberately keeps the operation provider-scoped; it does not reuse
the text-to-speech payload, where ``voice`` means a preset voice.
"""

from collections.abc import Mapping
from typing import Final, TypedDict

import httpx

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import extract_file_data
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import FileTypes


class VoiceCloneResponse(TypedDict):
    """Normalized response returned after a successful voice clone."""

    file_id: str
    voice_id: str
    model: str


class MinimaxVoiceCloneError(BaseLLMException):
    """Error raised when MiniMax rejects an upload or clone request."""

    def __init__(self, message: str, status_code: int = 0, headers: Mapping[str, str] | None = None) -> None:
        super().__init__(message=message, status_code=status_code, headers=headers)


class MinimaxVoiceCloneConfig:
    """Build and parse MiniMax voice-cloning requests."""

    GLOBAL_BASE_URL: Final[str] = "https://api.minimax.io"
    CN_BASE_URL: Final[str] = "https://api.minimaxi.com"
    FILE_UPLOAD_PATH: Final[str] = "/v1/files/upload"
    VOICE_CLONE_PATH: Final[str] = "/v1/voice_clone"
    UPLOAD_PURPOSES: Final[frozenset[str]] = frozenset({"voice_clone", "prompt_audio"})
    SUPPORTED_MODELS: Final[frozenset[str]] = frozenset(
        {"speech-2.8-hd", "speech-2.6-hd", "speech-02-hd", "speech-01-hd"}
    )

    @classmethod
    def _base_url(cls, api_base: str | None) -> str:
        """Return a regional host without a trailing ``/v1`` path."""
        base = (api_base or cls.GLOBAL_BASE_URL).rstrip("/")
        base = base.removesuffix("/v1")
        return base.rstrip("/")

    @classmethod
    def get_complete_url(cls, api_base: str | None = None, operation: str = "clone") -> str:
        """Return the endpoint URL for ``upload`` or ``clone``."""
        try:
            path = {"upload": cls.FILE_UPLOAD_PATH, "clone": cls.VOICE_CLONE_PATH}[operation]
        except KeyError as exc:
            raise ValueError("operation must be 'upload' or 'clone'") from exc
        return f"{cls._base_url(api_base)}{path}"

    @staticmethod
    def validate_environment(headers: dict[str, str] | None = None, api_key: str | None = None) -> dict[str, str]:
        """Add MiniMax authentication while preserving caller headers."""
        resolved_key = api_key or litellm.api_key or get_secret_str("MINIMAX_API_KEY")
        if not resolved_key:
            raise ValueError("MiniMax API key is required for voice cloning")

        result = dict(headers or {})
        result["Authorization"] = f"Bearer {resolved_key}"
        return result

    @classmethod
    def transform_upload_request(
        cls,
        file: FileTypes,
        purpose: str = "voice_clone",
    ) -> tuple[dict[str, tuple[str, bytes, str]], dict[str, str]]:
        """Convert a LiteLLM file value to MiniMax multipart fields."""
        if purpose not in cls.UPLOAD_PURPOSES:
            allowed = ", ".join(sorted(cls.UPLOAD_PURPOSES))
            raise ValueError(f"MiniMax file upload purpose must be one of: {allowed}")

        extracted = extract_file_data(file)
        filename = extracted.get("filename") or "voice-sample"
        content = extracted.get("content")
        if not isinstance(content, bytes):
            content = bytes(content)
        content_type = extracted.get("content_type") or "application/octet-stream"
        return {"file": (filename, content, content_type)}, {"purpose": purpose}

    @classmethod
    def transform_upload_response(cls, raw_response: httpx.Response) -> str:
        """Extract MiniMax's uploaded ``file_id`` and check its status code."""
        payload = cls._json_response(raw_response)
        cls._raise_for_api_status(payload, raw_response.status_code)
        file_id = cls._first_string(
            payload,
            ("file_id", "id"),
            nested=("file", "data", "result"),
        )
        if not file_id:
            raise MinimaxVoiceCloneError("MiniMax upload response did not include file_id", raw_response.status_code)
        return file_id

    @classmethod
    def transform_clone_request(cls, file_id: str, voice_id: str, model: str) -> dict[str, str]:
        """Build the required MiniMax voice-clone JSON body."""
        values = {"file_id": file_id, "voice_id": voice_id, "model": model}
        for name, value in values.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} is required for MiniMax voice cloning")
        if model not in cls.SUPPORTED_MODELS:
            supported = ", ".join(sorted(cls.SUPPORTED_MODELS))
            raise ValueError(f"Unsupported MiniMax voice-clone model {model!r}; supported models: {supported}")
        return {name: value.strip() for name, value in values.items()}

    @classmethod
    def transform_clone_response(
        cls,
        raw_response: httpx.Response,
        *,
        file_id: str,
        model: str,
    ) -> VoiceCloneResponse:
        """Normalize MiniMax's clone response to its generated ``voice_id``."""
        payload = cls._json_response(raw_response)
        cls._raise_for_api_status(payload, raw_response.status_code)
        voice_id = cls._first_string(
            payload,
            ("voice_id",),
            nested=("data", "result", "voice"),
        )
        if not voice_id:
            raise MinimaxVoiceCloneError("MiniMax clone response did not include voice_id", raw_response.status_code)
        return {"file_id": file_id, "voice_id": voice_id, "model": model}

    @staticmethod
    def _json_response(raw_response: httpx.Response) -> dict[str, object]:
        try:
            payload = raw_response.json()
        except ValueError as exc:
            raise MinimaxVoiceCloneError(
                "MiniMax returned a non-JSON voice-clone response",
                raw_response.status_code,
                raw_response.headers,
            ) from exc
        if not isinstance(payload, dict):
            raise MinimaxVoiceCloneError(
                "MiniMax returned an invalid voice-clone response",
                raw_response.status_code,
                raw_response.headers,
            )
        return payload

    @staticmethod
    def _first_string(
        payload: Mapping[str, object],
        keys: tuple[str, ...],
        nested: tuple[str, ...],
    ) -> str | None:
        for key in keys:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for key in nested:
            child = payload.get(key)
            if isinstance(child, Mapping):
                value = MinimaxVoiceCloneConfig._first_string(child, keys, nested=())
                if value:
                    return value
        return None

    @staticmethod
    def _raise_for_api_status(payload: Mapping[str, object], http_status: int) -> None:
        base_resp = payload.get("base_resp")
        status_code = base_resp.get("status_code") if isinstance(base_resp, Mapping) else None
        if http_status >= 400 or (status_code is not None and str(status_code) not in {"0", "200"}):
            status_message = base_resp.get("status_msg") if isinstance(base_resp, Mapping) else None
            detail = str(status_message or payload.get("message") or "MiniMax voice-clone request failed")
            raise MinimaxVoiceCloneError(detail, http_status or int(status_code or 0))
