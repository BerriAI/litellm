"""
Support for Oruk audio transcription via ``POST /v1/audio/transcriptions``.

API reference: https://oruk.ai/docs
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError
from typing_extensions import ReadOnly, TypedDict

from litellm.exceptions import AuthenticationError, BadRequestError, UnsupportedParamsError
from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, OpenAIAudioTranscriptionOptionalParams
from litellm.types.utils import FileTypes, TranscriptionResponse

_DEFAULT_API_BASE: Final = "https://speech-api.oruk.ai/v1"
_SUPPORTED_FORMATS: Final = frozenset(("json", "verbose_json", "text"))
_ENGLISH_LANGUAGES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "en": "en",
        "en-us": "en-US",
        "en-gb": "en-GB",
    }
)


class _OrukTranscript(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow", hide_input_in_errors=True)

    text: str
    language: str | None = None
    duration: float | None = None
    usage: Mapping[str, object] | None = None


class _ResponseFields(TypedDict, total=False):
    language: ReadOnly[str]
    duration: ReadOnly[float]
    usage: ReadOnly[Mapping[str, object]]


def _normalize_language(language: object, model: str, drop_params: bool) -> str | None:
    if language is None:
        return None
    if not isinstance(language, str):
        if drop_params:
            return None
        raise UnsupportedParamsError(
            message="Oruk transcription requires `language` to be a string ('en', 'en-US', or 'en-GB').",
            model=model,
            llm_provider="oruk",
        )
    normalized: Final = _ENGLISH_LANGUAGES.get(language.strip().lower())
    if normalized is None:
        if drop_params:
            return None
        raise UnsupportedParamsError(
            message=f"Oruk Fourier 2B supports English languages ('en', 'en-US', 'en-GB'), got language={language!r}. "
            "To drop unsupported params, set `drop_params=True`.",
            model=model,
            llm_provider="oruk",
        )
    return normalized


def _caller_headers(headers: Mapping[str, object]) -> Mapping[str, str]:
    return {
        name: value if isinstance(value, str) else json.dumps(value)
        for name, value in headers.items()
        if value is not None and name.lower() != "authorization"
    }


def _response_fields(payload: _OrukTranscript) -> _ResponseFields:
    language: Final[_ResponseFields] = {} if payload.language is None else {"language": payload.language}
    duration: Final[_ResponseFields] = {} if payload.duration is None else {"duration": payload.duration}
    usage: Final[_ResponseFields] = {} if payload.usage is None else {"usage": payload.usage}
    return {**language, **duration, **usage}


class OrukAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIAudioTranscriptionOptionalParams]:  # mutable-ok: public capability API returns a list
        return [
            "language",
            "prompt",
            "response_format",
        ]

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: dict[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: shared mapper returns provider params dict
        mapped: Final[dict[str, object]] = dict(optional_params)
        for key, value in non_default_params.items():
            if key == "response_format":
                if isinstance(value, str) and value in _SUPPORTED_FORMATS:
                    mapped[key] = value
                elif not drop_params:
                    raise UnsupportedParamsError(
                        message=f"Oruk transcription supports response_format in {sorted(_SUPPORTED_FORMATS)}, "
                        f"got {value!r}. To drop unsupported params, set `drop_params=True`.",
                        model=model,
                        llm_provider="oruk",
                    )
            elif key == "language":
                normalized = _normalize_language(value, model=model, drop_params=drop_params)
                if normalized is not None:
                    mapped[key] = normalized
            elif key == "prompt" and value is not None:
                mapped[key] = str(value)
        return mapped

    def validate_environment(
        self,
        headers: Mapping[str, object],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, object]:  # mutable-ok: shared HTTP handler expects a mutable header dict
        key: Final = api_key or get_secret_str("ORUK_API_KEY")
        if not key:
            raise AuthenticationError(
                message="Set ORUK_API_KEY or pass api_key for Oruk transcription",
                llm_provider="oruk",
                model=model,
            )
        extra_headers: Final = as_str_mapping(optional_params.get("extra_headers")) or {}
        return {
            **_caller_headers({**headers, **extra_headers}),
            "Authorization": f"Bearer {key}",
        }

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        return join_api_base(api_base or get_secret_str("ORUK_API_BASE") or _DEFAULT_API_BASE, "audio/transcriptions")

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
    ) -> AudioTranscriptionRequestData:
        try:
            audio: Final = process_audio_file(audio_file)
        except (ValueError, OSError) as exc:
            raise BadRequestError(message="Invalid audio upload", model=model, llm_provider="oruk") from exc
        if not audio.file_content:
            raise BadRequestError(
                message="Oruk transcription requires a non-empty audio file",
                model=model,
                llm_provider="oruk",
            )

        form_fields: Final[dict[str, object]] = {"model": model}
        for key in ("language", "prompt", "response_format"):
            value = optional_params.get(key)
            if value is not None:
                form_fields[key] = str(value)

        return AudioTranscriptionRequestData(
            data=form_fields,
            files={"file": (Path(audio.filename).name, audio.file_content, audio.content_type)},
        )

    def transform_audio_transcription_response(
        self,
        raw_response: httpx.Response,
    ) -> TranscriptionResponse:
        content_type: Final = raw_response.headers.get("content-type", "").lower()
        if content_type.startswith("text/plain"):
            return TranscriptionResponse(text=raw_response.text)

        try:
            payload: Final = _OrukTranscript.model_validate_json(raw_response.content)
        except ValidationError:
            raise BaseLLMException(
                status_code=502,
                message="Invalid Oruk transcription response",
                headers=raw_response.headers,
            ) from None

        metadata: Final = payload.model_dump(exclude={"text"}, exclude_none=True)
        response: Final = TranscriptionResponse(text=payload.text, **_response_fields(payload))
        response.hidden_params = metadata
        return response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Mapping[str, object] | httpx.Headers
    ) -> BaseLLMException:
        return BaseLLMException(status_code=status_code, message=error_message, headers=dict(headers))
