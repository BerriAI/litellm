import base64
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict

from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.rust_bridge import runtime
from litellm.rust_bridge.catalog import Route, RouteContext
from litellm.rust_bridge.host.timeouts import timeout_to_seconds
from litellm.rust_bridge.public_call import NativeCall
from litellm.rust_bridge.transcription.native import (
    NATIVE_ATRANSCRIPTION,
    NATIVE_TRANSCRIPTION,
    RustAtranscription,
    RustTranscription,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import FileTypes, TranscriptionResponse, TranscriptionUsageTokensObject

SUPPORTED_OPENAI_TRANSCRIPTION_PARAMS: Final = ("language", "prompt", "temperature", "response_format")
TEXT_RESPONSE_FORMATS: Final = frozenset({"json", "text"})


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number: Final = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _rejection(key: str, value: object) -> str | None:
    if key not in SUPPORTED_OPENAI_TRANSCRIPTION_PARAMS:
        return f"bedrock transcription does not support {key}."
    if key == "response_format" and value not in TEXT_RESPONSE_FORMATS:
        return f"bedrock transcription returns only the transcript text, so response_format={value!r} is not supported."
    if key == "temperature" and _as_float(value) is None:
        return f"bedrock transcription needs a number for temperature, got {value!r}."
    return None


@dataclass(frozen=True, slots=True)
class UnsupportedTranscriptionParams:
    message: str


def map_openai_transcription_params(
    non_default_params: Mapping[str, object], drop_params: bool
) -> Mapping[str, object] | UnsupportedTranscriptionParams:
    rejections: Final = {
        key: reason for key, value in non_default_params.items() if (reason := _rejection(key, value)) is not None
    }
    if rejections and not drop_params:
        return UnsupportedTranscriptionParams(
            message=f"{' '.join(rejections.values())} To drop it from the call, set `litellm.drop_params = True`."
        )
    return {
        key: _as_float(value) if key == "temperature" else value
        for key, value in non_default_params.items()
        if key not in rejections
    }


class _RustTranscriptionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    usage: TranscriptionUsageTokensObject | None = None


def _transcription_response(
    result: Mapping[str, object], optional_params: Mapping[str, object]
) -> TranscriptionResponse:
    parsed: Final = _RustTranscriptionResult.model_validate(result)
    response: Final = TranscriptionResponse(text=parsed.text).model_copy(update={"usage": parsed.usage})
    region_name: Final = (
        optional_params.get("aws_region_name") or get_secret_str("AWS_REGION_NAME") or get_secret_str("AWS_REGION")
    )
    response.hidden_params = {**response.hidden_params, "region_name": region_name}
    return response


class BedrockAudioTranscriptionRustDispatch:
    @staticmethod
    def _audio_payload(audio_file: FileTypes) -> dict[str, object]:
        processed_audio: Final = process_audio_file(audio_file)
        formats: Final = {
            "audio/flac": "flac",
            "audio/mpeg": "mp3",
            "audio/mp3": "mp3",
            "audio/ogg": "ogg",
            "audio/wav": "wav",
            "audio/x-wav": "wav",
        }
        audio_format: Final = formats.get(processed_audio.content_type) or (
            processed_audio.filename.rsplit(".", 1)[-1].lower() if "." in processed_audio.filename else ""
        )
        if audio_format not in {"wav", "mp3", "flac", "ogg"}:
            raise ValueError(f"Unsupported Bedrock audio format for file {processed_audio.filename!r}")
        return {
            "data": base64.b64encode(processed_audio.file_content).decode("ascii"),
            "format": audio_format,
            "filename": processed_audio.filename,
        }

    def audio_transcriptions(
        self,
        *,
        model: str,
        audio_file: FileTypes,
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str,
        extra_headers: dict[str, object] | None,
        optional_params: dict[str, object],
        timeout: float | httpx.Timeout | None,
    ) -> TranscriptionResponse:
        def native(rust: RustTranscription) -> TranscriptionResponse:
            fields: Final = {
                "model": model,
                "audio": self._audio_payload(audio_file),
                "api_key": api_key,
                "api_base": api_base,
                "custom_llm_provider": custom_llm_provider,
                "extra_headers": extra_headers,
                "optional_params": optional_params,
                "timeout_seconds": timeout_to_seconds(timeout),
            }
            call: Final = NativeCall(args=(), kwargs=fields, base={})
            return _transcription_response(rust(call), optional_params)

        return runtime.run(
            RouteContext(Route.TRANSCRIPTION, provider=custom_llm_provider, model=model),
            binding=NATIVE_TRANSCRIPTION,
            native=native,
            python=runtime.NO_PYTHON,
        )

    async def async_audio_transcriptions(
        self,
        *,
        model: str,
        audio_file: FileTypes,
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str,
        extra_headers: dict[str, object] | None,
        optional_params: dict[str, object],
        timeout: float | httpx.Timeout | None,
    ) -> TranscriptionResponse:
        async def native(rust: RustAtranscription) -> TranscriptionResponse:
            fields: Final = {
                "model": model,
                "audio": self._audio_payload(audio_file),
                "api_key": api_key,
                "api_base": api_base,
                "custom_llm_provider": custom_llm_provider,
                "extra_headers": extra_headers,
                "optional_params": optional_params,
                "timeout_seconds": timeout_to_seconds(timeout),
            }
            call: Final = NativeCall(args=(), kwargs=fields, base={})
            return _transcription_response(await rust(call), optional_params)

        return await runtime.arun(
            RouteContext(Route.TRANSCRIPTION, provider=custom_llm_provider, model=model),
            binding=NATIVE_ATRANSCRIPTION,
            native=native,
            python=runtime.NO_PYTHON,
        )
