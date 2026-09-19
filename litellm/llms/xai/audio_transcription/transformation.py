"""
Translates from OpenAI's `/v1/audio/transcriptions` to xAI's `/v1/stt`
"""

from collections.abc import Iterable, Mapping
from typing import Final, cast

from httpx import Headers, Response
from pydantic import BaseModel, ConfigDict

import litellm
from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIAudioTranscriptionOptionalParams,
)
from litellm.types.utils import FileTypes, TranscriptionResponse

from ...base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from ..common_utils import XAIModelInfo


class XAIAudioTranscriptionError(BaseLLMException):
    pass


class _XAISttWord(BaseModel):
    model_config = ConfigDict(extra="allow")
    text: str = ""
    start: float = 0.0
    end: float = 0.0
    speaker: str | None = None


class _XAISttResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    text: str = ""
    language: str = "unknown"
    duration: float | None = None
    words: list[_XAISttWord] | None = None


def _serialize_form_value(value: object) -> str | list[str]:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return [str(item) for item in cast(Iterable[object], value)]
    return str(value)


class XAIAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    @property
    def custom_llm_provider(self) -> str:
        return litellm.LlmProviders.XAI.value

    def get_supported_openai_params(self, model: str) -> list[OpenAIAudioTranscriptionOptionalParams]:
        return ["language"]

    def map_openai_params(
        self,
        non_default_params: dict[str, object],
        optional_params: dict[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        supported_params: Final = self.get_supported_openai_params(model)
        for k, v in non_default_params.items():
            if k in supported_params:
                optional_params[k] = v
        return optional_params

    def get_error_class(
        self, error_message: str, status_code: int, headers: dict[str, object] | Headers
    ) -> BaseLLMException:
        return XAIAudioTranscriptionError(message=error_message, status_code=status_code, headers=headers)

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
    ) -> AudioTranscriptionRequestData:
        processed_audio: Final = process_audio_file(audio_file)

        # Provider kwargs land in `extra_body` for openai_compatible_providers
        extra_body: Final = optional_params.get("extra_body")
        flat_params: Final[dict[str, object]] = {
            **(dict(cast(Mapping[str, object], extra_body)) if isinstance(extra_body, Mapping) else {}),
            **{k: v for k, v in optional_params.items() if k != "extra_body"},
        }

        openai_params: Final = self.get_supported_openai_params(model)
        excluded_params: Final = frozenset({"model", "OPENAI_TRANSCRIPTION_PARAMS", *openai_params})
        provider_specific_params: Final[dict[str, object]] = {
            k: v for k, v in flat_params.items() if v is not None and k not in excluded_params
        }

        form_data: Final[dict[str, str | list[str]]] = {"model": model}
        for key, value in provider_specific_params.items():
            form_data[key] = _serialize_form_value(value)
        for key in openai_params:
            value = flat_params.get(key)
            if value is not None:
                form_data[key] = _serialize_form_value(value)

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
        try:
            payload: Final = _XAISttResponse.model_validate_json(raw_response.content)
        except Exception as e:
            raise XAIAudioTranscriptionError(
                message=f"Error parsing xAI response: {e}",
                status_code=raw_response.status_code,
                headers=dict(raw_response.headers),
            )

        response: Final = TranscriptionResponse(text=payload.text)
        response["task"] = "transcribe"
        response["language"] = payload.language

        if payload.duration is not None:
            response["duration"] = payload.duration

        if payload.words is not None:
            response["words"] = [
                {
                    "word": word.text,
                    "start": word.start,
                    "end": word.end,
                    **({"speaker": word.speaker} if word.speaker is not None else {}),
                }
                for word in payload.words
            ]

        hidden_params: Final[dict[str, object]] = dict(payload.model_dump(mode="json"))
        if payload.duration is not None:
            hidden_params["audio_transcription_duration"] = payload.duration
        response._hidden_params = hidden_params  # pyright: ignore[reportPrivateUsage]  # TranscriptionResponse exposes no public hidden-params setter

        return response

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        stream: bool | None = None,
    ) -> str:
        base: Final = (XAIModelInfo.get_api_base(api_base) or "").rstrip("/")
        normalized: Final = base.removesuffix("/v1")
        return f"{normalized}/v1/stt"

    def validate_environment(
        self,
        headers: dict[str, object],
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, object]:
        resolved_key: Final = XAIModelInfo.get_api_key(api_key)
        if resolved_key is None:
            raise ValueError("xAI API key is required. Set XAI_API_KEY environment variable.")

        headers["Authorization"] = f"Bearer {resolved_key}"
        return headers
