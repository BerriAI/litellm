"""
Support for OpenAI's `/v1/audio/transcriptions` endpoint on deAPI. Every `response_format`, plain `text`
included, reports the real per-request cost in the `X-DeAPI-Cost` header.

Docs: https://docs.deapi.ai
"""

from collections.abc import Mapping
from typing import Final

import httpx
from pydantic import TypeAdapter

from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.litellm_core_utils.core_helpers import set_response_cost_in_hidden_params
from litellm.llms.base_llm.audio_transcription.transformation import AudioTranscriptionRequestData
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.transcriptions.whisper_transformation import OpenAIWhisperAudioTranscriptionConfig
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import FileTypes, TranscriptionResponse

from ..common_utils import DeapiException, authorized_headers, endpoint_url, reported_cost

_OPTIONAL_MAPPING: Final[TypeAdapter[Mapping[str, object] | None]] = TypeAdapter(Mapping[str, object] | None)


def _form_fields(model: str, optional_params: Mapping[str, object]) -> dict[str, object]:  # mutable-ok: httpx form data
    """LiteLLM parks non-OpenAI params, `model` included, under `extra_body` for the OpenAI SDK; a
    multipart body carries them as top-level text fields instead."""
    extras: Final = _OPTIONAL_MAPPING.validate_python(optional_params.get("extra_body"))
    nested: Final = extras.items() if extras is not None else ()
    fields: Final = (*optional_params.items(), *nested, ("model", model))
    return {key: value for key, value in fields if key != "extra_body"}


class DeapiAudioTranscriptionConfig(OpenAIWhisperAudioTranscriptionConfig):
    @property
    def has_native_transcription_endpoint(self) -> bool:
        return True

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict[str, object],  # mutable-ok: inherited contract
        litellm_params: dict[str, object],  # mutable-ok: inherited contract
        stream: bool | None = None,
    ) -> str:
        return endpoint_url(api_base, "audio/transcriptions")

    def validate_environment(
        self,
        headers: dict[str, object],  # mutable-ok: inherited contract
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: inherited contract
        optional_params: dict[str, object],  # mutable-ok: inherited contract
        litellm_params: dict[str, object],  # mutable-ok: inherited contract
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, object]:  # mutable-ok: inherited contract
        return authorized_headers(headers, api_key, model)

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict[str, object],  # mutable-ok: inherited contract
        litellm_params: dict[str, object],  # mutable-ok: inherited contract
    ) -> AudioTranscriptionRequestData:
        """The cost header comes with every format, so the Whisper default of `verbose_json` is not needed."""
        audio: Final = process_audio_file(audio_file)
        files: Final = {"file": (audio.filename, audio.file_content, audio.content_type)}
        return AudioTranscriptionRequestData(data=_form_fields(model, optional_params), files=files)

    def transform_audio_transcription_response(self, raw_response: httpx.Response) -> TranscriptionResponse:
        response: Final = super().transform_audio_transcription_response(raw_response)
        set_response_cost_in_hidden_params(response, reported_cost(raw_response))
        return response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, object] | httpx.Headers,  # mutable-ok: inherited contract
    ) -> BaseLLMException:
        return DeapiException(message=error_message, status_code=status_code, headers=headers)
