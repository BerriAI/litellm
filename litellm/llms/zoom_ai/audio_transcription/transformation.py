"""
Translates OpenAI's `/v1/audio/transcriptions` to Zoom Scribe Fast (`POST /scribe/transcribe`)

https://developers.zoom.us/docs/ai-services/scribe/fast-mode/
"""

import json
import re
from collections.abc import Mapping
from typing import Final

from httpx import Headers, Response
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm.constants import ZOOM_AI_DEFAULT_API_BASE
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.zoom_ai.common_utils import (
    ZOOM_AI_DEFAULT_LOCALES,
    ZoomAIException,
    zoom_ai_exception,
    zoom_ai_locale,
    zoom_ai_tracking_id,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIAudioTranscriptionOptionalParams,
)
from litellm.types.utils import FileTypes, TranscriptionResponse, TranscriptionUsageDurationObject

SUPPORTED_RESPONSE_FORMATS: Final = ("json",)
ZOOM_AI_BOOL_CONFIG_PARAMS: Final = frozenset(
    {"diarization", "word_time_offsets", "channel_separation", "profanity_filter"}
)
ZOOM_AI_STR_CONFIG_PARAMS: Final = frozenset({"segmentation_mode", "vocabulary_id"})
ZOOM_AI_CONFIG_PARAMS: Final = (
    ZOOM_AI_BOOL_CONFIG_PARAMS | ZOOM_AI_STR_CONFIG_PARAMS | {"vocabulary", "language_detection"}
)
# Models documented with continuous language detection, which is what callers expect from a multilingual model
CONTINUOUS_DETECTION_MODELS: Final = frozenset({"zoom-scribe-multilingual", "whisper"})

_LANGUAGE_MODEL: Final = re.compile(r"zoom-scribe-(?P<language>[a-z]{2,3})(?:-pro)?")
_FORM_BOOL: Final = TypeAdapter(bool)
_PHRASES: Final = TypeAdapter(tuple[str, ...])
_DETECTION: Final = TypeAdapter(dict[str, object])


class _ScribeResult(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    text_display: str = ""
    language: str | None = None


class _ScribeUsage(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    input_units: float
    unit_type: str


class _ScribeResponse(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    result: _ScribeResult
    duration_sec: float = 0.0
    usage: _ScribeUsage | None = None

    def billed_usage(self) -> TranscriptionUsageDurationObject | None:
        if self.usage is None or self.usage.unit_type != "seconds":
            return None
        return TranscriptionUsageDurationObject(type="duration", seconds=self.usage.input_units)


def _bool_param(value: object) -> bool:
    return _FORM_BOOL.validate_python(value.strip() if isinstance(value, str) else value)


def _vocabulary(value: object) -> Mapping[str, object]:
    """The OpenAI SDK sends a list as repeated form fields, so a one-phrase list reaches the proxy as a string."""
    phrases: Final = (value,) if isinstance(value, str) else _PHRASES.validate_python(value)
    return {"phrases": [phrase.strip() for phrase in phrases if phrase.strip()]}


def _language_detection(value: object) -> object:
    if not isinstance(value, str):
        return _DETECTION.validate_python(value)
    if value.lstrip().startswith("{"):
        return _DETECTION.validate_json(value)
    return {"mode": value.strip()}


def _decoded_param(key: str, value: object) -> object:
    if key in ZOOM_AI_BOOL_CONFIG_PARAMS:
        return _bool_param(value)
    if key == "vocabulary":
        return _vocabulary(value)
    if key == "language_detection":
        return _language_detection(value)
    return value


def _config_param(key: str, value: object) -> object:
    """Multipart form fields reach the proxy as strings, so each option is decoded to the type Scribe expects."""
    try:
        return _decoded_param(key, value)
    except ValidationError:
        raise ZoomAIException(message=f"Invalid `{key}` for Zoom Scribe: {value!r}", status_code=400) from None


def _model_locale(model: str) -> str | None:
    match: Final = _LANGUAGE_MODEL.fullmatch(model)
    if match is None:
        return None
    return ZOOM_AI_DEFAULT_LOCALES.get(match["language"])


def zoom_ai_recognition_config(model: str, optional_params: Mapping[str, object]) -> Mapping[str, object]:
    language: Final = optional_params.get("language")
    locale: Final = zoom_ai_locale(language) if isinstance(language, str) and language.strip() else _model_locale(model)
    provider_config: Final = {
        key: _config_param(key, value)
        for key, value in optional_params.items()
        if key in ZOOM_AI_CONFIG_PARAMS and value is not None
    }
    detection: Final[Mapping[str, object]] = (
        {"language_detection": {"mode": "continuous"}}
        if model in CONTINUOUS_DETECTION_MODELS and "language_detection" not in provider_config
        else {}
    )
    return {
        "model": model,
        **({"language": locale} if locale is not None else {}),
        **detection,
        **provider_config,
    }


class ZoomAIAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    @property
    def custom_llm_provider(self) -> str:
        return litellm.LlmProviders.ZOOM_AI.value

    def get_supported_openai_params(
        self,
        model: str,
    ) -> list[  # mutable-ok: overrides the BaseAudioTranscriptionConfig signature
        OpenAIAudioTranscriptionOptionalParams
    ]:
        return ["language", "response_format"]

    def map_openai_params(
        self,
        non_default_params: dict[str, object],  # mutable-ok: overrides the BaseAudioTranscriptionConfig signature
        optional_params: dict[str, object],  # mutable-ok: overrides the BaseAudioTranscriptionConfig signature
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: overrides the BaseAudioTranscriptionConfig signature
        supported_params: Final = self.get_supported_openai_params(model)
        mapped: Final = {
            **optional_params,
            **{k: v for k, v in non_default_params.items() if k in supported_params},
        }
        response_format: Final = mapped.get("response_format")
        if response_format is None or response_format in SUPPORTED_RESPONSE_FORMATS:
            return mapped
        if drop_params or litellm.drop_params:
            return {k: v for k, v in mapped.items() if k != "response_format"}
        raise UnsupportedParamsError(
            status_code=400,
            message=(
                f"Zoom Scribe does not support response_format={response_format!r}. "
                f"Supported values: {', '.join(SUPPORTED_RESPONSE_FORMATS)}. "
                "To drop unsupported openai params from the call, set `litellm.drop_params = True`"
            ),
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | Headers,  # mutable-ok: overrides the BaseAudioTranscriptionConfig signature
    ) -> BaseLLMException:
        return zoom_ai_exception(error_message, status_code, headers)

    def validate_environment(
        self,
        headers: dict[str, object],  # mutable-ok: overrides the BaseAudioTranscriptionConfig signature
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: overrides the BaseAudioTranscriptionConfig signature
        optional_params: dict[str, object],  # mutable-ok: overrides the BaseAudioTranscriptionConfig signature
        litellm_params: dict[str, object],  # mutable-ok: overrides the BaseAudioTranscriptionConfig signature
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, object]:  # mutable-ok: overrides the BaseAudioTranscriptionConfig signature
        resolved_key: Final = api_key or get_secret_str("ZOOM_AI_API_KEY")
        if not resolved_key:
            raise ZoomAIException(
                message="Missing Zoom AI Services API key. Set ZOOM_AI_API_KEY or pass api_key=...",
                status_code=401,
            )
        return {
            **headers,
            "Authorization": f"Bearer {resolved_key}",
            "x-zm-trackingid": zoom_ai_tracking_id(),
        }

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict[str, object],  # mutable-ok: overrides the BaseAudioTranscriptionConfig signature
        litellm_params: dict[str, object],  # mutable-ok: overrides the BaseAudioTranscriptionConfig signature
        stream: bool | None = None,
    ) -> str:
        base: Final = api_base or get_secret_str("ZOOM_AI_API_BASE") or ZOOM_AI_DEFAULT_API_BASE
        return join_api_base(base, "scribe/transcribe")

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict[str, object],  # mutable-ok: overrides the BaseAudioTranscriptionConfig signature
        litellm_params: dict[str, object],  # mutable-ok: overrides the BaseAudioTranscriptionConfig signature
    ) -> AudioTranscriptionRequestData:
        processed_audio: Final = process_audio_file(audio_file)
        return AudioTranscriptionRequestData(
            data={"config": json.dumps(zoom_ai_recognition_config(model, optional_params))},
            files={
                "file": (
                    processed_audio.filename,
                    processed_audio.file_content,
                    processed_audio.content_type,
                )
            },
        )

    def transform_audio_transcription_response(
        self,
        raw_response: Response,
    ) -> TranscriptionResponse:
        try:
            scribe: Final = _ScribeResponse.model_validate_json(raw_response.content)
        except ValidationError:
            raise ZoomAIException(
                message="Zoom Scribe returned a response that is not a transcription",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            ) from None
        text: Final = scribe.result.text_display
        if scribe.result.language:
            return TranscriptionResponse(
                text=text,
                usage=scribe.billed_usage(),
                task="transcribe",
                language=scribe.result.language,
                duration=scribe.duration_sec,
            )
        return TranscriptionResponse(
            text=text, usage=scribe.billed_usage(), task="transcribe", duration=scribe.duration_sec
        )
