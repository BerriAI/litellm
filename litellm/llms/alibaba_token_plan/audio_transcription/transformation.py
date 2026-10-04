from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from pathlib import PurePath
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.llms.alibaba_token_plan.common_utils import IMAGE_ENDPOINT, get_native_api_url, validate_headers
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import AllMessageValues, OpenAIAudioTranscriptionOptionalParams
from litellm.types.utils import FileTypes, TranscriptionResponse, TranscriptionUsageDurationObject


class _TranscriptionOptions(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    language: str | None = None
    prompt: str | None = None
    response_format: str | None = None


class _TranscriptionOutput(BaseModel):
    text: str


class _TranscriptionUsage(BaseModel):
    duration: float


class _TranscriptionResult(BaseModel):
    output: _TranscriptionOutput | None = None
    usage: _TranscriptionUsage | None = None
    code: str | None = None
    message: str | None = None


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
        supported: Final = self.get_supported_openai_params(model)
        unsupported: Final = tuple(
            key
            for key, value in non_default_params.items()
            if key not in supported or (key == "response_format" and value not in (None, "json"))
        )
        if unsupported and not (drop_params or litellm.drop_params):
            raise litellm.UnsupportedParamsError(
                status_code=400,
                message=(
                    f"Alibaba Token Plan transcription does not support: {', '.join(unsupported)}. "
                    "The supported response_format is 'json'."
                ),
                model=model,
                llm_provider="alibaba_token_plan",
            )
        return {
            **optional_params,
            **{key: value for key, value in non_default_params.items() if key not in unsupported},
        }

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
        return validate_headers(dict(headers), api_key)

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        return get_native_api_url(api_base, IMAGE_ENDPOINT)

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
        extra_body: Final = TypeAdapter(Mapping[str, object]).validate_python(optional_params.get("extra_body") or {})
        options: Final = _TranscriptionOptions.model_validate(
            {
                **extra_body,
                **{key: value for key, value in optional_params.items() if key not in ("model", "extra_body")},
            }
        )
        if options.response_format not in (None, "json"):
            raise self.get_error_class("Alibaba Token Plan transcription only supports response_format='json'", 400, {})
        audio: Final = process_audio_file(audio_file)
        audio_format: Final = PurePath(audio.filename).suffix.lstrip(".").lower() or "wav"
        if audio_format not in (
            "aac",
            "amr",
            "avi",
            "flac",
            "flv",
            "m4a",
            "mkv",
            "mov",
            "mp3",
            "mp4",
            "mpeg",
            "ogg",
            "opus",
            "wav",
            "webm",
            "wma",
            "wmv",
        ):
            raise self.get_error_class(
                f"Unsupported Alibaba Token Plan transcription audio format: {audio_format}", 400, {}
            )
        audio_data: Final = f"data:{audio.content_type};base64,{base64.b64encode(audio.file_content).decode('ascii')}"
        context: Final = (
            [{"role": "user", "content": [{"type": "input_text", "text": options.prompt}]}] if options.prompt else []
        )
        return AudioTranscriptionRequestData(
            data={
                "model": model.removeprefix("alibaba_token_plan/"),
                "input": {
                    "messages": [
                        *context,
                        {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": audio_data}}]},
                    ]
                },
                "parameters": {
                    "format": audio_format,
                    **({"language_hints": [options.language]} if options.language else {}),
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
            raise self.get_error_class(
                "Invalid Alibaba Token Plan transcription response", 502, raw_response.headers
            ) from None
        if parsed.code or parsed.output is None:
            raise self.get_error_class(
                parsed.message or parsed.code or "Alibaba Token Plan transcription response is missing output",
                502,
                raw_response.headers,
            )
        response: Final = TranscriptionResponse(text=parsed.output.text)
        if parsed.usage is not None:
            response.usage = TranscriptionUsageDurationObject(type="duration", seconds=parsed.usage.duration)
            response["duration"] = parsed.usage.duration
        return response
