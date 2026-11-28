import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate
from types import MappingProxyType
from typing import Annotated, ClassVar, Final, Literal, TypeAlias
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
from httpx import Headers
from pydantic import BaseModel, ConfigDict, Discriminator, Tag, TypeAdapter, ValidationError

import litellm
from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import AllMessageValues, OpenAIAudioTranscriptionOptionalParams
from litellm.types.utils import FileTypes, TranscriptionResponse, TranscriptionUsageDurationObject

from ..common_utils import GradiumException


@dataclass(frozen=True, slots=True)
class _InputFormat:
    content_type: str
    bytes_per_second: int | None = None


_WAV: Final = "wav"
_OPUS: Final = "opus"
_INPUT_FORMATS: Final[Mapping[str, _InputFormat]] = MappingProxyType(
    {
        _WAV: _InputFormat("audio/wav"),
        _OPUS: _InputFormat("audio/ogg"),
        "pcm": _InputFormat("audio/pcm", 48_000),
        **{f"pcm_{rate}": _InputFormat("audio/pcm", rate * 2) for rate in (8000, 16000, 22050, 24000, 44100, 48000)},
        **{companded: _InputFormat("audio/pcm", 8000) for companded in ("ulaw_8000", "mulaw_8000", "alaw_8000")},
    }
)
_FORMATS_BY_EXTENSION: Final[Mapping[str, str]] = MappingProxyType(
    {"wav": _WAV, "pcm": "pcm", "ogg": _OPUS, "opus": _OPUS}
)


class _Text(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["text"]
    text: str
    start_s: float
    stream_id: int = 0


class _EndText(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["end_text"]
    stop_s: float
    stream_id: int = 0


class _Error(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["error"]
    message: str


class _EndOfStream(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: Literal["end_of_stream"]


class _Ignored(BaseModel):
    model_config = ConfigDict(frozen=True)


_KNOWN_KINDS: Final = frozenset({"text", "end_text", "error", "end_of_stream"})


def _message_kind(value: object) -> str:
    fields: Final = as_str_mapping(value)
    kind: Final = fields.get("type") if fields is not None else None
    return kind if isinstance(kind, str) and kind in _KNOWN_KINDS else "ignored"


_Message: TypeAlias = _Text | _EndText | _Error | _EndOfStream | _Ignored
_MESSAGE: Final[TypeAdapter[_Message]] = TypeAdapter(
    Annotated[
        Annotated[_Text, Tag("text")]
        | Annotated[_EndText, Tag("end_text")]
        | Annotated[_Error, Tag("error")]
        | Annotated[_EndOfStream, Tag("end_of_stream")]
        | Annotated[_Ignored, Tag("ignored")],
        Discriminator(_message_kind),
    ],
    config=ConfigDict(hide_input_in_errors=True),
)
_JSON_CONFIG: Final = TypeAdapter(dict[str, object])
_TEMPERATURE: Final = TypeAdapter(float)
_NO_STOPS: Final[Mapping[int, float]] = MappingProxyType({})


class _Segment(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    start: float
    end: float
    text: str


def _with_stop(later_stops: Mapping[int, float], message: _Message) -> Mapping[int, float]:
    if not isinstance(message, _EndText):
        return later_stops
    return MappingProxyType({**later_stops, message.stream_id: message.stop_s})


def _segments(messages: Sequence[_Message]) -> tuple[_Segment, ...]:
    stops_after: Final = tuple(accumulate(reversed(messages), _with_stop, initial=_NO_STOPS))
    texts: Final = tuple(
        (index, message)
        for index, message in enumerate(messages)
        if isinstance(message, _Text) and message.text.strip()
    )
    return tuple(
        _Segment(
            id=segment_id,
            start=text.start_s,
            end=stops_after[len(messages) - 1 - index].get(text.stream_id, text.start_s),
            text=text.text.strip(),
        )
        for segment_id, (index, text) in enumerate(texts)
    )


def _sniffed_format(audio: bytes) -> str | None:
    if audio[:4] == b"RIFF" and audio[8:12] == b"WAVE":
        return _WAV
    if audio[:4] == b"OggS":
        return _OPUS
    return None


def _extension(filename: str) -> str:
    return filename.rpartition(".")[2].lower() if "." in filename else ""


class GradiumAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    ASR_ENDPOINT_PATH = "/api/speech/asr"
    AUTO_DETECT_LANGUAGE = "any"
    RESPONSE_FORMATS: ClassVar[frozenset[str]] = frozenset({"json", "text", "verbose_json"})

    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIAudioTranscriptionOptionalParams]:  # mutable-ok: BaseAudioTranscriptionConfig declares list
        return ["language", "temperature", "response_format"]

    def map_openai_params(
        self,
        non_default_params: dict[str, object],  # mutable-ok: signature fixed by BaseAudioTranscriptionConfig
        optional_params: dict[str, object],  # mutable-ok: signature fixed by BaseAudioTranscriptionConfig
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: callers merge provider params into it downstream
        response_format: Final = non_default_params.get("response_format")
        unsupported_format: Final = response_format is not None and response_format not in self.RESPONSE_FORMATS
        if unsupported_format and not drop_params:
            raise litellm.UnsupportedParamsError(
                status_code=400,
                message=(
                    f"Gradium speech-to-text does not support response_format={response_format!r}. "
                    f"Supported values: {sorted(self.RESPONSE_FORMATS)}. Set drop_params to true to drop it."
                ),
                llm_provider="gradium",
                model=model,
            )
        return {
            **optional_params,
            **{
                key: value
                for key, value in non_default_params.items()
                if key in ("language", "temperature") or (key == "response_format" and not unsupported_format)
            },
        }

    def _json_config(self, model: str, optional_params: Mapping[str, object]) -> Mapping[str, object]:
        caller_config: Final = self._caller_json_config(model, optional_params)
        language: Final = optional_params.get("language") or caller_config.get("language") or self.AUTO_DETECT_LANGUAGE
        temperature: Final = optional_params.get("temperature")
        return {
            **caller_config,
            "language": language,
            **({"temp": self._temperature(model, temperature)} if temperature is not None else {}),
        }

    @staticmethod
    def _caller_json_config(model: str, optional_params: Mapping[str, object]) -> Mapping[str, object]:
        extra_body: Final = optional_params.get("extra_body")
        from_extra_body: Final = extra_body.get("json_config") if isinstance(extra_body, Mapping) else None
        raw: Final = from_extra_body if from_extra_body is not None else optional_params.get("json_config")
        if raw is None:
            return {}
        try:
            return (
                _JSON_CONFIG.validate_json(raw) if isinstance(raw, str | bytes) else _JSON_CONFIG.validate_python(raw)
            )
        except ValidationError as error:
            raise litellm.BadRequestError(
                message="Gradium json_config must be a JSON object.",
                model=model,
                llm_provider="gradium",
            ) from error

    @staticmethod
    def _temperature(model: str, temperature: object) -> float:
        try:
            return _TEMPERATURE.validate_python(temperature)
        except ValidationError as error:
            raise litellm.BadRequestError(
                message=f"Gradium temperature must be a number, got {temperature!r}.",
                model=model,
                llm_provider="gradium",
            ) from error

    @staticmethod
    def _requested_input_format(model: str, optional_params: Mapping[str, object]) -> str | None:
        requested: Final = optional_params.get("input_format")
        if requested is None:
            return None
        if isinstance(requested, str) and requested in _INPUT_FORMATS:
            return requested
        raise litellm.BadRequestError(
            message=f"Gradium input_format must be one of {sorted(_INPUT_FORMATS)}, got {requested!r}.",
            model=model,
            llm_provider="gradium",
        )

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict[str, object],  # mutable-ok: signature fixed by BaseAudioTranscriptionConfig
        litellm_params: dict[str, object],  # mutable-ok: signature fixed by BaseAudioTranscriptionConfig
        stream: bool | None = None,
    ) -> str:
        if not api_base:
            raise litellm.APIConnectionError(
                message="Gradium API base is required. Set `api_base` on the model, e.g. https://eu.api.gradium.ai.",
                llm_provider="gradium",
                model=model,
            )

        input_format: Final = self._requested_input_format(model, optional_params)
        query: Final = urlencode(
            {
                "model": model,
                "json_config": json.dumps(self._json_config(model, optional_params), separators=(",", ":")),
                **({"input_format": input_format} if input_format is not None else {}),
            }
        )
        root: Final = api_base.rstrip("/").removesuffix("/api")
        return f"{root}{self.ASR_ENDPOINT_PATH}?{query}"

    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: signature fixed by BaseAudioTranscriptionConfig
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: signature fixed by BaseAudioTranscriptionConfig
        optional_params: dict[str, object],  # mutable-ok: signature fixed by BaseAudioTranscriptionConfig
        litellm_params: dict[str, object],  # mutable-ok: signature fixed by BaseAudioTranscriptionConfig
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: the handler sets Content-Type on the returned headers
        if not api_key:
            raise litellm.AuthenticationError(
                message="Gradium API key is required. Set `api_key` on the model.",
                llm_provider="gradium",
                model=model,
            )

        return {**headers, "x-api-key": api_key}

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | Headers,  # mutable-ok: signature fixed by BaseAudioTranscriptionConfig
    ) -> BaseLLMException:
        return GradiumException(message=error_message, status_code=status_code, headers=headers)

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict[str, object],  # mutable-ok: signature fixed by BaseAudioTranscriptionConfig
        litellm_params: dict[str, object],  # mutable-ok: signature fixed by BaseAudioTranscriptionConfig
    ) -> AudioTranscriptionRequestData:
        audio: Final = process_audio_file(audio_file)
        input_format: Final = (
            self._requested_input_format(model, optional_params)
            or _sniffed_format(audio.file_content)
            or _FORMATS_BY_EXTENSION.get(_extension(audio.filename))
        )
        if input_format is None:
            raise litellm.BadRequestError(
                message=(
                    f"Gradium speech-to-text cannot read {audio.filename!r}. Send WAV, Ogg Opus or raw PCM audio, "
                    f"or set input_format to one of {sorted(_INPUT_FORMATS)}."
                ),
                model=model,
                llm_provider="gradium",
            )
        return AudioTranscriptionRequestData(
            data=audio.file_content, content_type=_INPUT_FORMATS[input_format].content_type
        )

    def transform_audio_transcription_response(self, raw_response: httpx.Response) -> TranscriptionResponse:
        try:
            messages: Final = tuple(
                _MESSAGE.validate_json(line) for line in raw_response.text.split("\n") if line.strip()
            )
        except ValidationError as error:
            raise GradiumException(
                message=f"Gradium answered a malformed transcription stream: {error}",
                status_code=502,
                headers=raw_response.headers,
            ) from error

        failure: Final = next((message for message in messages if isinstance(message, _Error)), None)
        if failure is not None:
            raise GradiumException(message=failure.message, status_code=502, headers=raw_response.headers)
        if not any(isinstance(message, _EndOfStream) for message in messages):
            raise GradiumException(
                message=(
                    "Gradium closed the transcription stream before end_of_stream. "
                    "Check that the language is supported and that the audio matches its declared format."
                ),
                status_code=502,
                headers=raw_response.headers,
            )

        segments: Final = _segments(messages)
        raw_seconds: Final = self._raw_audio_seconds(raw_response.request)
        return TranscriptionResponse(
            text=" ".join(segment.text for segment in segments),
            segments=tuple(segment.model_dump() for segment in segments),
            usage=(
                TranscriptionUsageDurationObject(type="duration", seconds=raw_seconds)
                if raw_seconds is not None
                else None
            ),
        )

    @staticmethod
    def _raw_audio_seconds(request: httpx.Request) -> float | None:
        requested: Final = parse_qs(urlsplit(str(request.url)).query).get("input_format", ())
        input_format: Final = (
            requested[0] if requested else "pcm" if "audio/pcm" in request.headers.get_list("content-type") else None
        )
        bytes_per_second: Final = _INPUT_FORMATS[input_format].bytes_per_second if input_format is not None else None
        return len(request.content) / bytes_per_second if bytes_per_second else None
