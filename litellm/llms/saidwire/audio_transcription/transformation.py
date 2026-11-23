"""
Saidwire speech-to-text through its OpenAI-compatible `/v1/audio/transcriptions` endpoint.

API reference: https://saidwire.com/docs
"""

from collections.abc import Mapping, Sequence
from typing import Final

import httpx
from pydantic import ConfigDict, TypeAdapter

from litellm.constants import SAIDWIRE_DEFAULT_API_BASE
from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIAudioTranscriptionOptionalParams,
)
from litellm.types.utils import FileTypes, TranscriptionResponse

_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_PASSTHROUGH_RESPONSE_FIELDS: Final = ("task", "language", "duration", "segments", "words")


class SaidwireAudioTranscriptionException(BaseLLMException):
    pass


class SaidwireAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIAudioTranscriptionOptionalParams]:  # mutable-ok: BaseAudioTranscriptionConfig contract
        return [
            "language",
            "prompt",
            "response_format",
            "temperature",
            "timestamp_granularities",
        ]

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: dict[str, object],  # mutable-ok: BaseAudioTranscriptionConfig contract
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: BaseAudioTranscriptionConfig contract
        supported_params: Final = self.get_supported_openai_params(model)
        optional_params.update({k: v for k, v in non_default_params.items() if k in supported_params})
        return optional_params

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        base_url: Final = api_base or get_secret_str("SAIDWIRE_API_BASE") or SAIDWIRE_DEFAULT_API_BASE
        return join_api_base(base_url, "audio/transcriptions")

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | httpx.Headers,  # mutable-ok: BaseAudioTranscriptionConfig contract
    ) -> BaseLLMException:
        return SaidwireAudioTranscriptionException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, object]:  # mutable-ok: BaseAudioTranscriptionConfig contract
        resolved_key: Final = api_key or get_secret_str("SAIDWIRE_API_KEY")
        if not resolved_key:
            raise SaidwireAudioTranscriptionException(
                message="Saidwire API key not found. Pass `api_key=...` or set the SAIDWIRE_API_KEY environment variable.",
                status_code=401,
                headers={},
            )
        return {
            **(headers or {}),
            "Authorization": f"Bearer {resolved_key}",
            "accept": "application/json",
        }

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
    ) -> AudioTranscriptionRequestData:
        processed_audio: Final = process_audio_file(audio_file)
        form_fields: Final[dict[str, object]] = {"model": model}  # mutable-ok: filled below, then handed to httpx
        for key in self.get_supported_openai_params(model):
            value = optional_params.get(key)
            if value is None:
                continue
            if key == "timestamp_granularities":
                form_fields["timestamp_granularities[]"] = value
            else:
                form_fields[key] = value

        files: Final = {
            "file": (
                processed_audio.filename,
                processed_audio.file_content,
                processed_audio.content_type,
            )
        }
        return AudioTranscriptionRequestData(data=form_fields, files=files)

    def transform_audio_transcription_response(
        self,
        raw_response: httpx.Response,
    ) -> TranscriptionResponse:
        content_type: Final = (raw_response.headers.get("content-type") or "").lower()
        if "application/json" not in content_type:
            return TranscriptionResponse(text=raw_response.text)

        try:
            payload: Final = _JSON_OBJECT.validate_python(raw_response.json())
        except Exception as e:
            raise SaidwireAudioTranscriptionException(
                message="Received a malformed JSON body from Saidwire audio transcription",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            ) from e

        text: Final = payload.get("text")
        usage: Final = payload.get("usage")
        response: Final = TranscriptionResponse(
            text=text if isinstance(text, str) else "",
            usage=_JSON_OBJECT.validate_python(usage) if isinstance(usage, Mapping) else None,
        )
        for field in _PASSTHROUGH_RESPONSE_FIELDS:
            if field in payload:
                response[field] = payload[field]
        response.hidden_params = dict(payload)
        return response
