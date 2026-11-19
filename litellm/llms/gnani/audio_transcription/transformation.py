import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, OpenAIAudioTranscriptionOptionalParams
from litellm.types.utils import FileTypes, TranscriptionResponse

_SHARED_FIELDS: Final = frozenset(("format", "enable_substitution", "substitution_map", "bias_list"))
_V25_FIELDS: Final = _SHARED_FIELDS | {"bias_score", "itn_native_numerals"}
_V3_FIELDS: Final = _SHARED_FIELDS | {"keyterms", "topic", "previous_text", "speaker_names"}


class _PrismaTranscript(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, hide_input_in_errors=True)

    success: Literal[True]
    transcript: str
    request_id: str | None = None
    detected_language_code: str | None = None
    truncated: bool | None = None


def _normalized_hint(hint: str) -> str:
    if hint.lower() == "indic-auto":
        return "indic-auto"
    return f"{hint.lower()}-IN" if len(hint) == 2 else hint


def _language_code(model: str, language: object) -> str:
    if language is not None and not isinstance(language, str):
        raise UnsupportedParamsError(message="language must be a string", model=model, llm_provider="gnani")
    hints: Final = tuple(filter(None, (hint.strip() for hint in (language or "").split(","))))
    if not hints and model == "prisma-v3.0":
        return "indic-auto"
    if model == "prisma-v2.5" and (len(hints) != 1 or hints[0].lower() == "indic-auto"):
        raise UnsupportedParamsError(
            message="Prisma v2.5 requires one explicit language, for example language='hi'.",
            model=model,
            llm_provider="gnani",
        )
    return ",".join(_normalized_hint(hint) for hint in hints)


class GnaniAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIAudioTranscriptionOptionalParams]:  # mutable-ok: the public capability API returns a list
        return ["language", "response_format"]

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: the shared mapper adds provider parameters to this dict
        if non_default_params.get("response_format", "json") != "json" and not drop_params:
            raise UnsupportedParamsError(
                message="Gnani transcription supports response_format=json", model=model, llm_provider="gnani"
            )
        return {**optional_params, "language_code": _language_code(model, non_default_params.get("language"))}

    def validate_environment(
        self,
        headers: Mapping[str, object],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, object]:  # mutable-ok: the HTTP handler accepts and updates a header dict
        key: Final = api_key or get_secret_str("GNANI_API_KEY")
        if not key:
            raise BaseLLMException(401, "Set GNANI_API_KEY or pass api_key for Gnani transcription")
        return {**headers, "X-API-Key-ID": key}

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        return join_api_base(api_base or get_secret_str("GNANI_API_BASE") or "https://api.vachana.ai", "stt/v3")

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
    ) -> AudioTranscriptionRequestData:
        if model not in ("prisma-v2.5", "prisma-v3.0"):
            raise BaseLLMException(400, "Gnani REST transcription supports prisma-v2.5 and prisma-v3.0")
        try:
            audio: Final = process_audio_file(audio_file)
        except (ValueError, OSError) as exc:
            raise BaseLLMException(400, "Invalid audio upload") from exc
        if not audio.file_content:
            raise BaseLLMException(400, "Gnani transcription requires a non-empty audio file")
        fields: Final = {**(as_str_mapping(optional_params.get("extra_body")) or {}), **optional_params}
        supported: Final = _V25_FIELDS if model == "prisma-v2.5" else _V3_FIELDS
        incompatible: Final = (_V25_FIELDS | _V3_FIELDS) - supported
        unsupported: Final = sorted(key for key in incompatible if fields.get(key) is not None)
        if unsupported:
            raise BaseLLMException(400, f"{model} does not support {', '.join(unsupported)}")
        form: Final = {
            key: value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            for key, value in fields.items()
            if key in supported and value is not None
        }
        return AudioTranscriptionRequestData(
            data={**form, "model": model, "language_code": optional_params["language_code"]},
            files={"audio_file": (Path(audio.filename).name, audio.file_content, audio.content_type)},
        )

    def transform_audio_transcription_response(self, raw_response: httpx.Response) -> TranscriptionResponse:
        try:
            payload: Final = _PrismaTranscript.model_validate_json(raw_response.content)
        except ValidationError:
            raise BaseLLMException(502, "Invalid Gnani transcription response", headers=raw_response.headers) from None
        response: Final = TranscriptionResponse(text=payload.transcript)
        response.hidden_params = payload.model_dump(exclude={"success", "transcript"}, exclude_none=True)
        return response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Mapping[str, object] | httpx.Headers
    ) -> BaseLLMException:
        return BaseLLMException(status_code, error_message, headers=dict(headers))
