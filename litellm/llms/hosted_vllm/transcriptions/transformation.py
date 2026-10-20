"""
Transformation logic for Hosted VLLM audio transcription
"""

from collections.abc import Mapping
from typing import Final

import httpx
from pydantic import TypeAdapter

from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.transcriptions.whisper_transformation import (
    OpenAIWhisperAudioTranscriptionConfig,
)
from litellm.types.utils import FileTypes


class HostedVLLMAudioTranscriptionError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: dict | httpx.Headers | None = None,
    ):
        super().__init__(status_code=status_code, message=message, headers=headers)


_OBJECT_TUPLE: Final = TypeAdapter(tuple[object, ...])
_STRING_OBJECT_DICT: Final = TypeAdapter(dict[str, object])


def _form_value(
    value: object,
) -> str | list[str]:  # mutable-ok: httpx multipart data takes list values for repeated form fields
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return [str(item) for item in _OBJECT_TUPLE.validate_python(value)]
    return str(value)


class HostedVLLMAudioTranscriptionConfig(OpenAIWhisperAudioTranscriptionConfig):
    def __init__(self) -> None:
        pass

    @property
    def has_native_transcription_endpoint(self) -> bool:
        """Send through the shared http handler, which honors per-request `ssl_verify`
        (e.g. a custom CA bundle); the OpenAI SDK transport only uses the global setting."""
        return True

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        if not api_base:
            raise ValueError("api_base must be provided for Hosted VLLM audio transcription")
        trimmed: Final = api_base.rstrip("/")
        if trimmed.endswith("/v1/audio/transcriptions"):
            return trimmed
        if trimmed.endswith("/v1"):
            return f"{trimmed}/audio/transcriptions"
        return f"{trimmed}/v1/audio/transcriptions"

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
    ) -> AudioTranscriptionRequestData:
        """Multipart form, as vLLM's OpenAI-compatible /v1/audio/transcriptions expects:
        the audio under `files`, `extra_body` flattened into the other form fields."""
        audio: Final = process_audio_file(audio_file)
        explicit_type: Final = audio_file[2] if isinstance(audio_file, tuple) and len(audio_file) >= 3 else None
        content_type: Final = explicit_type if isinstance(explicit_type, str) and explicit_type else audio.content_type
        params: Final = _STRING_OBJECT_DICT.validate_python(optional_params)
        extra_body: Final = params.get("extra_body")
        nested: Final[Mapping[str, object]] = (
            _STRING_OBJECT_DICT.validate_python(extra_body) if isinstance(extra_body, Mapping) else {}
        )
        fields: Final[Mapping[str, object]] = {
            **nested,
            **{key: value for key, value in params.items() if key != "extra_body"},
        }
        data: Final = {
            "model": model,
            **{key: _form_value(value) for key, value in fields.items() if value is not None and key != "model"},
        }
        return AudioTranscriptionRequestData(
            data=data,
            files={"file": (audio.filename, audio.file_content, content_type)},
        )
