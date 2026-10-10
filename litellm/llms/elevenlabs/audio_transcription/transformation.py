"""
Translates from OpenAI's `/v1/audio/transcriptions` to ElevenLabs's `/v1/speech-to-text`
"""

import math
from collections.abc import Iterable, Mapping
from typing import Final

from httpx import Headers, Response
from pydantic import ConfigDict, TypeAdapter

import litellm
from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIAudioTranscriptionOptionalParams,
)
from litellm.types.utils import FileTypes, TranscriptionResponse

from ...base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from ..common_utils import ElevenLabsException

_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_JSON_OBJECTS: Final = TypeAdapter(Iterable[Mapping[str, object]], config=ConfigDict(hide_input_in_errors=True))
_FORM_OPENAI_PARAMS: Final = frozenset({"language", "temperature"})


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        number: Final = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _timestamp(value: object) -> float:
    seconds: Final = _finite_float(value)
    if seconds is None or seconds < 0 or _finite_float(seconds * 1000) is None:
        return 0.0
    return seconds


def _time_span(item: Mapping[str, object]) -> Mapping[str, float]:
    start: Final = _timestamp(item.get("start"))
    return {"start": start, "end": max(start, _timestamp(item.get("end")))}


def _speaker_fields(item: Mapping[str, object]) -> Mapping[str, str]:
    speaker: Final = item.get("speaker_id")
    return {"speaker": speaker} if isinstance(speaker, str) else {}


def _word_fields(item: Mapping[str, object]) -> Mapping[str, object]:
    logprob: Final = _finite_float(item.get("logprob"))
    return {
        "word": item.get("text", ""),
        **_time_span(item),
        **({} if logprob is None else {"logprob": logprob}),
        **_speaker_fields(item),
    }


def _audio_event_fields(item: Mapping[str, object]) -> Mapping[str, object]:
    return {
        "text": item.get("text", ""),
        **_time_span(item),
        **_speaker_fields(item),
    }


class ElevenLabsAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    @property
    def custom_llm_provider(self) -> str:
        return litellm.LlmProviders.ELEVENLABS.value

    def get_supported_openai_params(self, model: str) -> list[OpenAIAudioTranscriptionOptionalParams]:
        return ["language", "temperature", "response_format"]

    @property
    def supports_subtitle_synthesis(self) -> bool:
        return True

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params: Final = self.get_supported_openai_params(model)
        for k, v in non_default_params.items():
            if k in supported_params:
                if k == "language":
                    # Map OpenAI language format to ElevenLabs language_code
                    optional_params["language_code"] = v
                else:
                    optional_params[k] = v
        return optional_params

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

        # Prepare form data
        form_data: Final = {"model_id": model}

        #########################################################
        # Add OpenAI Compatible Parameters
        #########################################################
        for key, value in optional_params.items():
            if key in _FORM_OPENAI_PARAMS and value is not None:
                # Convert values to strings for form data, but skip None values
                form_data[key] = str(value)

        #########################################################
        # Add Provider Specific Parameters
        #########################################################
        provider_specific_params: Final = self.get_provider_specific_params(
            model=model,
            optional_params=optional_params,
            openai_params=self.get_supported_openai_params(model),
        )

        for key, value in provider_specific_params.items():
            form_data[key] = str(value)
        #########################################################
        #########################################################

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
        Transforms the raw response from ElevenLabs to the TranscriptionResponse format
        """
        try:
            response_json: Final = raw_response.json()
            response_object: Final = _JSON_OBJECT.validate_python(response_json)

            # Extract the main transcript text
            text: Final = response_object.get("text", "")

            # Create TranscriptionResponse object
            response: Final = TranscriptionResponse(text=text)

            # Add additional metadata matching OpenAI format
            response["task"] = "transcribe"
            response["language"] = response_object.get("language_code", "unknown")

            if "words" in response_object:
                items: Final = tuple(_JSON_OBJECTS.validate_python(response_object["words"]))
                response["words"] = [_word_fields(item) for item in items if item.get("type") == "word"]
                response["audio_events"] = [
                    _audio_event_fields(item) for item in items if item.get("type") == "audio_event"
                ]

            language_probability: Final = _finite_float(response_object.get("language_probability"))
            if language_probability is not None:
                response["language_probability"] = language_probability

            duration: Final = _finite_float(response_object.get("audio_duration_secs"))
            if duration is not None and duration >= 0:
                response["duration"] = duration

            # Store full response in hidden params
            response.hidden_params = response_json

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
        api_base = api_base.rstrip("/")  # Remove trailing slash if present

        # ElevenLabs speech-to-text endpoint
        url: Final = f"{api_base}/v1/speech-to-text"

        return url

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
