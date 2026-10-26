"""
Translates from OpenAI's `/v1/audio/transcriptions` to ElevenLabs's `/v1/speech-to-text`
"""

import io
from collections.abc import Iterable, Iterator, Mapping
from itertools import chain, groupby
from types import MappingProxyType
from typing import Final, Protocol

from httpx import Headers, Response
from pydantic import ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.elevenlabs import (
    ElevenLabsSTTChunk,
    ElevenLabsSTTMultichannelResponse,
    ElevenLabsSTTWord,
    OpenAIDiarizedSegment,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIAudioTranscriptionOptionalParams,
)
from litellm.types.utils import FileTypes, TranscriptionResponse, TranscriptionUsageDurationObject

from ...base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from ..common_utils import ElevenLabsException

_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_JSON_OBJECTS: Final = TypeAdapter(Iterable[Mapping[str, object]], config=ConfigDict(hide_input_in_errors=True))
_HIDDEN_PARAMS: Final = TypeAdapter(dict[str, object])
_CHANNELS: Final = TypeAdapter(int)
_DIARIZATION_KEYS: Final = frozenset(("diarize", "use_multi_channel", "multichannel_output_style"))
# OpenAI parameter names that ElevenLabs spells differently
_ELEVENLABS_PARAM_NAMES: Final = MappingProxyType({"language": "language_code"})


class _JsonResponse(Protocol):
    def json(self) -> object: ...


def _json_body(response: _JsonResponse) -> object:
    return response.json()


class ElevenLabsAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    @property
    def custom_llm_provider(self) -> str:
        return litellm.LlmProviders.ELEVENLABS.value

    def get_supported_openai_params(self, model: str) -> list[OpenAIAudioTranscriptionOptionalParams]:
        return ["language", "temperature", "response_format"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params: Final = self.get_supported_openai_params(model)
        params: Final = _JSON_OBJECT.validate_python(non_default_params)
        mapped_params: Final = {
            _ELEVENLABS_PARAM_NAMES.get(k, k): v
            for k, v in params.items()
            if k in supported_params and k != "response_format"
        }
        # ElevenLabs always returns JSON; the only response_format that changes the
        # request is diarized_json, which maps to diarization. Other formats are
        # shaped from the same JSON at response time.
        diarize_params: Final = {"diarize": True} if params.get("response_format") == "diarized_json" else {}
        return {**optional_params, **mapped_params, **diarize_params}

    def get_error_class(self, error_message: str, status_code: int, headers: dict | Headers) -> BaseLLMException:
        return ElevenLabsException(message=error_message, status_code=status_code, headers=headers)

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
    ) -> AudioTranscriptionRequestData:
        """
        Transforms the audio transcription request for ElevenLabs API.

        Returns AudioTranscriptionRequestData with both form data and files.

        Returns:
            AudioTranscriptionRequestData: Structured data with form data and files
        """

        # Use common utility to process the audio file
        processed_audio: Final = process_audio_file(audio_file)
        supported_params: Final = self.get_supported_openai_params(model)

        provider_specific_params: Final = _resolve_diarization_params(
            self.get_provider_specific_params(
                model=model,
                optional_params=optional_params,
                openai_params=supported_params,
            ),
            processed_audio.file_content,
        )

        openai_params: Final = {
            key: str(value) for key, value in optional_params.items() if key in supported_params and value is not None
        }
        form_data: Final = {
            "model_id": model,
            **openai_params,
            **{key: str(value) for key, value in provider_specific_params.items()},
        }

        # Prepare files
        files: Final = {
            "file": (
                processed_audio.filename,
                processed_audio.file_content,
                processed_audio.content_type,
            )
        }

        return AudioTranscriptionRequestData(data=form_data, files=files)

    def transform_audio_transcription_response(
        self,
        raw_response: Response,
    ) -> TranscriptionResponse:
        """
        Transforms the raw ElevenLabs speech-to-text response.

        When the response carries speaker information (diarize=true) or per-channel
        transcripts (use_multi_channel=true), it is shaped into OpenAI's
        diarized_json form (segments with speaker labels + usage). Otherwise it
        falls back to a flat transcript with word-level timestamps.
        """
        try:
            response_json: Final = _json_body(raw_response)
            response: Final = _transcription_response(_JSON_OBJECT.validate_python(response_json))

            # Store full response in hidden params
            response.hidden_params = _HIDDEN_PARAMS.validate_python(response_json)

            return response

        except Exception as e:
            raise ValueError(f"Error transforming ElevenLabs response: {e}\nResponse: {raw_response.text}")

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        if api_base is None:
            api_base = get_secret_str("ELEVENLABS_API_BASE") or "https://api.elevenlabs.io"
        return join_api_base(api_base, "v1/speech-to-text")

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        api_key = api_key or get_secret_str("ELEVENLABS_API_KEY")
        if api_key is None:
            raise ValueError("ElevenLabs API key is required. Set ELEVENLABS_API_KEY environment variable.")

        auth_header: Final = {
            "xi-api-key": api_key,
        }

        headers.update(auth_header)
        return headers


def _truthy(value: object) -> bool:
    return str(value).strip().lower() == "true"


def _channel_count(audio_content: bytes) -> int:
    """Best-effort channel count from the audio header; 1 if it can't be read."""
    try:
        import soundfile
    except ImportError:
        return 1

    try:
        channels: Final = soundfile.info(io.BytesIO(audio_content)).channels  # pyright: ignore[reportUnknownMemberType]  # soundfile ships no type hints
        return _CHANNELS.validate_python(channels)
    except (RuntimeError, TypeError, ValidationError):
        return 1


def _resolve_diarization_params(provider_params: Mapping[str, object], audio_content: bytes) -> Mapping[str, object]:
    """
    ElevenLabs rejects diarize and use_multi_channel together. Diarization
    requested on multi-channel audio uses the channel as the speaker; on mono it
    falls back to acoustic diarization. An explicit use_multi_channel from the
    caller wins over auto-detection, and an unreadable header falls back to mono.
    """
    diarize_requested: Final = _truthy(provider_params.get("diarize"))
    multichannel: Final = (
        _truthy(provider_params["use_multi_channel"])
        if "use_multi_channel" in provider_params
        else diarize_requested and _channel_count(audio_content) > 1
    )
    others: Final = {key: value for key, value in provider_params.items() if key not in _DIARIZATION_KEYS}

    if multichannel:
        return {
            **others,
            "use_multi_channel": True,
            "multichannel_output_style": provider_params.get("multichannel_output_style", "combined"),
        }
    if diarize_requested:
        return {**others, "diarize": True}
    return others


def _transcription_response(response_object: Mapping[str, object]) -> TranscriptionResponse:
    """
    The audio-transcription response transformer only sees the provider body, so
    the output shape is decided from it: per-channel transcripts or any word with
    a speaker id produce diarized segments, anything else stays a flat transcript.
    """
    if "transcripts" in response_object:
        return _diarized_response(_combine_channels(ElevenLabsSTTMultichannelResponse.model_validate(response_object)))

    words: Final = tuple(_JSON_OBJECTS.validate_python(response_object.get("words", ())))
    if any(word.get("speaker_id") is not None for word in words):
        return _diarized_response(ElevenLabsSTTChunk.model_validate(response_object))
    return _flat_response(response_object, words)


def _flat_response(
    response_object: Mapping[str, object], words: tuple[Mapping[str, object], ...]
) -> TranscriptionResponse:
    response: Final = TranscriptionResponse(text=response_object.get("text", ""))

    # Add additional metadata matching OpenAI format
    response["task"] = "transcribe"
    response["language"] = response_object.get("language_code", "unknown")

    # Map ElevenLabs words to OpenAI format
    if "words" in response_object:
        response["words"] = [
            {
                "word": word_data.get("text", ""),
                "start": word_data.get("start", 0),
                "end": word_data.get("end", 0),
            }
            for word_data in words
            if word_data.get("type") == "word"
        ]

    return response


def _diarized_response(transcript: ElevenLabsSTTChunk) -> TranscriptionResponse:
    response: Final = TranscriptionResponse(text=transcript.text)
    response["task"] = "transcribe"
    response["language"] = transcript.language_code or "unknown"
    response["segments"] = _build_segments(transcript.words)
    if transcript.audio_duration_secs is not None:
        response["duration"] = transcript.audio_duration_secs
        response["usage"] = TranscriptionUsageDurationObject(type="duration", seconds=transcript.audio_duration_secs)
    return response


def _combine_channels(parsed: ElevenLabsSTTMultichannelResponse) -> ElevenLabsSTTChunk:
    """
    Flatten per-channel transcripts into a single time-ordered word stream, using
    the channel as the speaker so the two sides of e.g. a stereo call don't collide
    on the per-channel speaker ids ElevenLabs assigns independently.
    """
    words: Final = chain.from_iterable(
        _channel_words(position, transcript) for position, transcript in enumerate(parsed.transcripts)
    )
    return ElevenLabsSTTChunk(
        text=" ".join(transcript.text for transcript in parsed.transcripts if transcript.text),
        language_code=next(
            (transcript.language_code for transcript in parsed.transcripts if transcript.language_code), None
        ),
        words=tuple(sorted(words, key=lambda word: word.start if word.start is not None else 0.0)),
        audio_duration_secs=parsed.audio_duration_secs,
    )


def _channel_words(position: int, transcript: ElevenLabsSTTChunk) -> Iterator[ElevenLabsSTTWord]:
    """
    The list position is the fallback channel when channel_index is absent, so
    distinct channels never collapse onto the same speaker.
    """
    channel: Final = transcript.channel_index if transcript.channel_index is not None else position
    return (word.model_copy(update={"speaker_id": f"speaker_{channel}"}) for word in transcript.words)


def _build_segments(words: tuple[ElevenLabsSTTWord, ...]) -> tuple[OpenAIDiarizedSegment, ...]:
    spoken: Final = (word for word in words if word.type == "word" and word.start is not None)
    groups: Final = (tuple(group) for _, group in groupby(spoken, key=lambda word: word.speaker_id))
    return tuple(
        OpenAIDiarizedSegment(
            id=f"segment_{index}",
            start=group[0].start if group[0].start is not None else 0.0,
            end=group[-1].end if group[-1].end is not None else 0.0,
            speaker=group[0].speaker_id or "speaker_0",
            text=" ".join(word.text for word in group),
            type="transcript.text.segment",
        )
        for index, group in enumerate(groups)
    )
