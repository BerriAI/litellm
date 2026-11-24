"""
Translates from OpenAI's `/v1/audio/transcriptions` to ElevenLabs's `/v1/speech-to-text`
"""

import json
from collections.abc import Iterable, Mapping
from contextlib import suppress
from typing import Final, Protocol

from httpx import Headers, Response
from pydantic import ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.litellm_core_utils.url_utils import join_api_base
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
_HIDDEN_PARAMS: Final = TypeAdapter(dict[str, object])
_FORM_ITEMS: Final = TypeAdapter(tuple[object, ...])
_CLIENT_BLOCKED_FIELDS: Final = frozenset({"model_id", "extra_body"})


class _JsonResponse(Protocol):
    def json(self) -> object: ...


def _json_body(response: _JsonResponse) -> object:
    return response.json()


def _json_compatible(value: object) -> object:
    if isinstance(value, Mapping):
        return dict(value.items())
    return str(value)


def _json_text(value: object) -> str:
    return json.dumps(value, default=_json_compatible)


def _extra_body_fields(extra_body: object) -> Mapping[str, object]:
    with suppress(ValidationError):
        return _JSON_OBJECT.validate_python(extra_body if isinstance(extra_body, Mapping) else {})
    raise ElevenLabsException(status_code=400, message="ElevenLabs extra_body keys must be strings")


def _form_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Mapping):
        return _json_text(value)
    return str(value)


def _encoded_form_value(value: object) -> str | tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return _form_scalar(value)
    items: Final = tuple(item for item in _FORM_ITEMS.validate_python(value) if item is not None)
    if any(isinstance(item, (Mapping, list, tuple)) for item in items):
        return _json_text(items)
    return tuple(_form_scalar(item) for item in items)


def _form_value(value: object) -> str | tuple[str, ...]:
    with suppress(TypeError, ValueError, RecursionError):
        return _encoded_form_value(value)
    raise ElevenLabsException(status_code=400, message="ElevenLabs form value cannot be encoded as JSON or text")


class ElevenLabsAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    @property
    def custom_llm_provider(self) -> str:
        return litellm.LlmProviders.ELEVENLABS.value

    def get_supported_openai_params(self, model: str) -> list[OpenAIAudioTranscriptionOptionalParams]:
        return ["language", "temperature"]

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

        openai_params: Final = self.get_supported_openai_params(model)
        params: Final = _JSON_OBJECT.validate_python(optional_params)
        provider_specific_params: Final = _JSON_OBJECT.validate_python(
            self.get_provider_specific_params(
                model=model,
                optional_params=optional_params,
                openai_params=openai_params,
            )
        )
        extra_fields: Final = _extra_body_fields(provider_specific_params.get("extra_body"))
        form_data: Final = {
            "model_id": model,
            **{
                key: _form_value(value)
                for key, value in extra_fields.items()
                if key not in _CLIENT_BLOCKED_FIELDS and value is not None
            },
            **{key: _form_value(value) for key, value in params.items() if key in openai_params and value is not None},
            **{
                key: _form_value(value)
                for key, value in provider_specific_params.items()
                if key not in _CLIENT_BLOCKED_FIELDS
            },
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
        Transforms the raw response from ElevenLabs to the TranscriptionResponse format
        """
        try:
            response_json: Final = _json_body(raw_response)
            response_object: Final = _JSON_OBJECT.validate_python(response_json)

            # Extract the main transcript text
            text: Final = response_object.get("text", "")

            # Create TranscriptionResponse object
            response: Final = TranscriptionResponse(text=text)

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
                    for word_data in _JSON_OBJECTS.validate_python(response_object["words"])
                    if word_data.get("type") == "word"
                ]

            # Store full response in hidden params
            response.hidden_params = _HIDDEN_PARAMS.validate_python(response_json)

            return response

        except Exception as e:
            raise ValueError(f"Error transforming ElevenLabs response: {e}") from None

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
