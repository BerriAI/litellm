from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from pathlib import PurePath
from typing import Final

import httpx
from pydantic import BaseModel, ValidationError

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.llms.alibaba_token_plan.common_utils import IMAGE_PATH, get_api_url, validate_headers
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import AllMessageValues, OpenAIAudioTranscriptionOptionalParams
from litellm.types.utils import FileTypes, TranscriptionResponse, TranscriptionUsageDurationObject

SUPPORTED_RESPONSE_FORMATS: Final = ("json", "text")


class _TranscriptionOutput(BaseModel, frozen=True):
    text: str


class _TranscriptionUsage(BaseModel, frozen=True):
    duration: float


class _TranscriptionResult(BaseModel, frozen=True):
    output: _TranscriptionOutput
    usage: _TranscriptionUsage | None = None


class AlibabaTokenPlanAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    @property
    def has_native_transcription_endpoint(self) -> bool:
        return True

    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIAudioTranscriptionOptionalParams]:  # mutable-ok: base config requires a list
        return ["language", "prompt", "response_format"]

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: the base config contract returns mutable request parameters
        response_format: Final = non_default_params.get("response_format")
        unsupported: Final = response_format is not None and response_format not in SUPPORTED_RESPONSE_FORMATS
        if unsupported and not (drop_params or litellm.drop_params):
            raise UnsupportedParamsError(
                status_code=400,
                message=(
                    f"Alibaba Token Plan transcription does not support response_format={response_format!r}. "
                    f"Supported values: {', '.join(SUPPORTED_RESPONSE_FORMATS)}. "
                    "To drop unsupported openai params from the call, set `litellm.drop_params = True`"
                ),
            )
        return {**optional_params, **{k: v for k, v in non_default_params.items() if k in ("language", "prompt")}}

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: the HTTP handler adds headers after validation
        return validate_headers(headers, api_key)

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        return get_api_url(api_base, IMAGE_PATH)

    def get_error_class(
        self, error_message: str, status_code: int, headers: Mapping[str, str] | httpx.Headers
    ) -> BaseLLMException:
        return BaseLLMException(message=error_message, status_code=status_code, headers=dict(headers))

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
    ) -> AudioTranscriptionRequestData:
        audio: Final = process_audio_file(audio_file)
        audio_data: Final = f"data:{audio.content_type};base64,{base64.b64encode(audio.file_content).decode('ascii')}"
        prompt: Final = optional_params.get("prompt")
        language: Final = optional_params.get("language")
        context: Final = [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}] if prompt else []
        return AudioTranscriptionRequestData(
            data={
                "model": model,
                "input": {
                    "messages": [
                        *context,
                        {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": audio_data}}]},
                    ]
                },
                "parameters": {
                    "format": PurePath(audio.filename).suffix.lstrip(".").lower() or "wav",
                    **({"language_hints": [language]} if language else {}),
                },
            },
            content_type="application/json",
        )

    def transform_audio_transcription_response(self, raw_response: httpx.Response) -> TranscriptionResponse:
        if not raw_response.is_success:
            raise self.get_error_class(raw_response.text, raw_response.status_code, raw_response.headers)
        try:
            parsed: Final = _TranscriptionResult.model_validate_json(raw_response.content)
        except ValidationError:
            raise self.get_error_class(raw_response.text, 502, raw_response.headers) from None
        response: Final = TranscriptionResponse(text=parsed.output.text)
        if parsed.usage is not None:
            response.usage = TranscriptionUsageDurationObject(type="duration", seconds=parsed.usage.duration)
            response["duration"] = parsed.usage.duration
        return response
