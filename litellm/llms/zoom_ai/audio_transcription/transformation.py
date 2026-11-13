"""
Translates OpenAI's `/v1/audio/transcriptions` to Zoom Scribe Fast (`POST /scribe/transcribe`)

https://developers.zoom.us/docs/ai-services/scribe/fast-mode/
"""

import json
import re
from collections.abc import Iterator, Mapping
from pathlib import PurePath
from typing import Final, Literal

from httpx import Headers, Response
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from typing_extensions import NotRequired, ReadOnly, TypedDict

import litellm
from litellm.constants import ZOOM_AI_DEFAULT_API_BASE
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.audio_utils.subtitle_utils import SUBTITLE_RESPONSE_FORMATS
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

SUPPORTED_RESPONSE_FORMATS: Final = ("json", "verbose_json", "diarized_json", "srt", "vtt")
ZOOM_AI_BOOL_CONFIG_PARAMS: Final = frozenset(
    {"diarization", "word_time_offsets", "channel_separation", "profanity_filter"}
)
POLISHING_FAILED_PREFIX: Final = "[polishing_failed:"
ZOOM_AI_STR_CONFIG_PARAMS: Final = frozenset({"segmentation_mode", "vocabulary_id"})
ZOOM_AI_CONFIG_PARAMS: Final = (
    ZOOM_AI_BOOL_CONFIG_PARAMS | ZOOM_AI_STR_CONFIG_PARAMS | {"vocabulary", "language_detection"}
)
# `zoom_ai/auto` leaves the model to Scribe, which picks one from the language and detection settings
ZOOM_AI_AUTO_MODEL: Final = "auto"
# The multilingual model detects the language continuously, which is what callers expect from it. whisper is left
# out because Scribe refuses continuous detection on it unless exactly one candidate language is named
CONTINUOUS_DETECTION_MODELS: Final = frozenset({"zoom-scribe-multilingual"})

_LANGUAGE_MODEL: Final = re.compile(r"zoom-scribe-(?P<language>[a-z]{2,3})(?:-pro(?:-[a-z0-9.]+)*)?")
_FORM_BOOL: Final = TypeAdapter(bool)
_PHRASES: Final = TypeAdapter(tuple[str, ...])
_DETECTION: Final = TypeAdapter(dict[str, object])


class _ScribeWord(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    word: str
    start: float
    end: float


class _ScribeSegment(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    id: str = ""
    text: str = ""
    start: float = 0.0
    end: float = 0.0
    channel: int = 0
    speaker: str = ""
    language: str | None = None
    words: tuple[_ScribeWord, ...] = ()


class _ScribeResult(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    text_display: str = ""
    language: str | None = None
    segments: tuple[_ScribeSegment, ...] = ()


class _Segment(TypedDict):
    id: ReadOnly[str]
    type: ReadOnly[Literal["transcript.text.segment"]]
    start: ReadOnly[float]
    end: ReadOnly[float]
    text: ReadOnly[str]
    channel: ReadOnly[int]
    speaker: NotRequired[ReadOnly[str]]
    language: NotRequired[ReadOnly[str]]


class _Word(TypedDict):
    word: ReadOnly[str]
    start: ReadOnly[float]
    end: ReadOnly[float]


class _OptionalFields(TypedDict, total=False):
    language: ReadOnly[str]
    segments: ReadOnly[list[_Segment]]  # writable-ok: never written, list is the OpenAI response shape
    words: ReadOnly[list[_Word]]  # writable-ok: never written, list is the OpenAI response shape
    polished_text: ReadOnly[str]


class _ScribeUsage(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    input_units: float
    unit_type: str


class _ScribePolishing(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    text: str = ""


class _ScribeOutputs(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    text_polishing: _ScribePolishing | None = None


class _ScribeResponse(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    result: _ScribeResult
    duration_sec: float = 0.0
    model: str | None = None
    usage: _ScribeUsage | None = None
    outputs: _ScribeOutputs | None = None

    def polished_text(self) -> str | None:
        """Scribe reports a failed polish as a `[polishing_failed: ...]` marker, which is not text to show."""
        polishing: Final = self.outputs.text_polishing if self.outputs is not None else None
        if polishing is None or not polishing.text or polishing.text.startswith(POLISHING_FAILED_PREFIX):
            return None
        return polishing.text

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
    if key in ZOOM_AI_BOOL_CONFIG_PARAMS or key == "text_polishing":
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
    response_format: Final = optional_params.get("response_format")
    granularities: Final = optional_params.get("timestamp_granularities")
    wants_words: Final = response_format in SUBTITLE_RESPONSE_FORMATS or (
        isinstance(granularities, (list, tuple)) and "word" in granularities
    )
    provider_config: Final = {
        **({"diarization": True} if response_format == "diarized_json" else {}),
        **({"word_time_offsets": True} if wants_words else {}),
        **{
            key: _config_param(key, value)
            for key, value in optional_params.items()
            if key in ZOOM_AI_CONFIG_PARAMS and value is not None
        },
    }
    detection: Final[Mapping[str, object]] = (
        {"language_detection": {"mode": "continuous"}}
        if model in CONTINUOUS_DETECTION_MODELS and "language_detection" not in provider_config
        else {}
    )
    return {
        **({"model": model} if model != ZOOM_AI_AUTO_MODEL else {}),
        **({"language": locale} if locale is not None else {}),
        **detection,
        **provider_config,
    }


class _SegmentLabels(TypedDict, total=False):
    speaker: ReadOnly[str]
    language: ReadOnly[str]


def _segment(index: int, segment: _ScribeSegment) -> _Segment:
    """One shape for every format: OpenAI's segment fields, plus Scribe's channel, speaker and language."""
    speaker: Final[_SegmentLabels] = {"speaker": segment.speaker} if segment.speaker else {}
    language: Final[_SegmentLabels] = {"language": segment.language} if segment.language else {}
    base: Final[_Segment] = {
        "id": segment.id or f"seg_{index}",
        "type": "transcript.text.segment",
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "channel": segment.channel,
    }
    labelled: Final[_Segment] = {**base, **speaker, **language}
    return labelled


def _words(segments: tuple[_ScribeSegment, ...]) -> list[_Word]:  # mutable-ok: OpenAI response shape
    return [_Word(word=word.word, start=word.start, end=word.end) for word in _flattened_words(segments)]


def _flattened_words(segments: tuple[_ScribeSegment, ...]) -> Iterator[_ScribeWord]:
    for segment in segments:
        yield from segment.words


def _optional_fields(scribe: _ScribeResponse) -> _OptionalFields:
    result: Final = scribe.result
    segments: Final = [_segment(index, segment) for index, segment in enumerate(result.segments)]
    words: Final = _words(result.segments)
    polished: Final = scribe.polished_text()
    language: Final[_OptionalFields] = {"language": result.language} if result.language else {}
    with_segments: Final[_OptionalFields] = {"segments": segments} if segments else {}
    with_words: Final[_OptionalFields] = {"words": words} if words else {}
    with_polished: Final[_OptionalFields] = {"polished_text": polished} if polished is not None else {}
    merged: Final[_OptionalFields] = {**language, **with_segments, **with_words, **with_polished}
    return merged


def zoom_ai_outputs(optional_params: Mapping[str, object]) -> Mapping[str, str]:
    """Text Polishing is requested in Scribe's top-level `outputs` field, next to `config`."""
    requested: Final = optional_params.get("text_polishing")
    if requested is None or not _config_param("text_polishing", requested):
        return {}
    return {"outputs": json.dumps({"text_polishing": {"enabled": True}})}


def _transcription_response(scribe: _ScribeResponse) -> TranscriptionResponse:
    return TranscriptionResponse(
        text=scribe.result.text_display,
        usage=scribe.billed_usage(),
        task="transcribe",
        duration=scribe.duration_sec,
        **_optional_fields(scribe),
    )


class ZoomAIAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    @property
    def custom_llm_provider(self) -> str:
        return litellm.LlmProviders.ZOOM_AI.value

    @property
    def supports_subtitle_synthesis(self) -> bool:
        return True

    def get_supported_openai_params(
        self,
        model: str,
    ) -> list[  # mutable-ok: overrides the BaseAudioTranscriptionConfig signature
        OpenAIAudioTranscriptionOptionalParams
    ]:
        return ["language", "response_format", "timestamp_granularities"]

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
            data={
                "config": json.dumps(zoom_ai_recognition_config(model, optional_params)),
                **zoom_ai_outputs(optional_params),
            },
            files={
                "file": (
                    PurePath(processed_audio.filename).name,
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
        response: Final = _transcription_response(scribe)
        if scribe.model:
            response.hidden_params = {**response.hidden_params, "provider_response_model": scribe.model}
        return response
