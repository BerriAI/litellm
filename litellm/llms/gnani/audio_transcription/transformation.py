import json
import re
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm.constants import GNANI_REJECTED_OPENAI_TRANSCRIPTION_PARAMS, OPENAI_TRANSCRIPTION_PARAMS
from litellm.exceptions import AuthenticationError, BadRequestError, UnsupportedParamsError
from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.litellm_core_utils.core_helpers import as_object_list, as_str_mapping
from litellm.litellm_core_utils.model_cost_table import published
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
_V25_MODEL: Final = "prisma-v2.5"
_V25_FIELDS_FLAG: Final = "gnani_v25_request_fields"
_NOT_FORWARDED: Final = frozenset(
    (
        *OPENAI_TRANSCRIPTION_PARAMS,
        *GNANI_REJECTED_OPENAI_TRANSCRIPTION_PARAMS,
        "prompt",
        "temperature",
        "extra_body",
        "extra_headers",
        "additional_drop_params",
    )
)
_REGION_CODE: Final = re.compile(r"([A-Za-z]{2})-([A-Za-z]{2})")
_STRING_LIST: Final = TypeAdapter(list[str])
_V25_RECOMMENDED_BIAS_SCORE: Final = 1.0
_KEY_HEADER: Final = "X-API-Key-ID"


class _PrismaTranscript(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, hide_input_in_errors=True)

    success: Literal[True]
    transcript: str
    request_id: str | None = None
    detected_language_code: str | None = None
    truncated: bool | None = None


def _is_v25(model: str) -> bool:
    name: Final = model.removeprefix("gnani/")
    if (as_str_mapping(published().get(f"gnani/{name}")) or {}).get("litellm_provider") != "gnani":
        return name == _V25_MODEL
    flags: Final = litellm.get_model_info(model=name, custom_llm_provider="gnani").get("provider_specific_entry")
    return name == _V25_MODEL if flags is None else flags.get(_V25_FIELDS_FLAG) == 1


def _normalized_hint(hint: str) -> str:
    if hint.lower() == "indic-auto":
        return "indic-auto"
    if len(hint) == 2:
        return f"{hint.lower()}-IN"
    region: Final = _REGION_CODE.fullmatch(hint)
    return f"{region[1].lower()}-{region[2].upper()}" if region else hint


def _language_code(model: str, language: object) -> str:
    if language is not None and not isinstance(language, str):
        raise UnsupportedParamsError(message="language must be a string", model=model, llm_provider="gnani")
    hints: Final = tuple(filter(None, (hint.strip() for hint in (language or "").split(","))))
    if not hints and not _is_v25(model):
        return "indic-auto"
    if _is_v25(model) and (len(hints) != 1 or hints[0].lower() == "indic-auto"):
        raise UnsupportedParamsError(
            message="Prisma v2.5 requires one explicit language, for example language='hi'.",
            model=model,
            llm_provider="gnani",
        )
    return ",".join(_normalized_hint(hint) for hint in hints)


def _is_blank(value: object) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _joined_languages(languages: object) -> object:
    try:
        return ",".join(_STRING_LIST.validate_python(languages))
    except ValidationError:
        return languages


def _language_hint(fields: Mapping[str, object]) -> object:
    hints: Final = (fields.get("language_code"), _joined_languages(fields.get("languages")), fields.get("language"))
    return next((hint for hint in hints if not _is_blank(hint)), None)


def _is_single_word(term: str) -> bool:
    return bool(term) and all(unicodedata.category(char)[0] in "LM" for char in term)


def _is_phrase(term: str) -> bool:
    return bool(term.strip())


def _accepted_keywords(model: str, keywords: object) -> tuple[str, ...] | None:
    try:
        terms: Final = tuple(_STRING_LIST.validate_python([keywords] if isinstance(keywords, str) else keywords))
    except ValidationError:
        return None
    accepts: Final = _is_single_word if _is_v25(model) else _is_phrase
    return terms if all(map(accepts, terms)) else None


def _rejected_keywords(model: str) -> UnsupportedParamsError:
    message: Final = (
        "Prisma v2.5 keywords must be single words made of letters only. "
        "Use gnani/prisma-v3.0 for phrases, digits or punctuation, or set drop_params=True to drop them."
        if _is_v25(model)
        else "Gnani keywords must be non-blank strings. Set drop_params=True to drop them."
    )
    return UnsupportedParamsError(message=message, model=model, llm_provider="gnani")


def _with_keywords_as_terms(fields: Mapping[str, object], model: str, drops_unsupported: bool) -> Mapping[str, object]:
    keywords: Final = fields.get("keywords")
    target: Final = "bias_list" if _is_v25(model) else "keyterms"
    if keywords is None or fields.get("bias_list") is not None or fields.get(target) is not None:
        return fields
    terms: Final = _accepted_keywords(model, keywords)
    if terms is None and drops_unsupported:
        return fields
    if terms is None:
        raise _rejected_keywords(model)
    if not terms:
        return fields
    if not _is_v25(model):
        return {**fields, "keyterms": terms}
    bias_score: Final = fields.get("bias_score")
    return {
        **fields,
        "bias_list": terms,
        "bias_score": _V25_RECOMMENDED_BIAS_SCORE if bias_score is None else bias_score,
    }


def _dropped_names(dropped: object) -> frozenset[str]:
    return frozenset(name for name in as_object_list(dropped) or () if isinstance(name, str))


def _without_names(fields: Mapping[str, object], names: frozenset[str]) -> Mapping[str, object]:
    return {key: value for key, value in fields.items() if key not in names}


def _as_text(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _caller_headers(headers: Mapping[str, object]) -> Mapping[str, str]:
    return {
        name: value if isinstance(value, str) else json.dumps(value)
        for name, value in headers.items()
        if value is not None and name.lower() != _KEY_HEADER.lower()
    }


def _with_prompt_as_previous_text(fields: Mapping[str, object]) -> Mapping[str, object]:
    prompt: Final = fields.get("prompt")
    if prompt is None or fields.get("previous_text") is not None:
        return fields
    return {**fields, "previous_text": prompt}


class _CallerFields(TypedDict, total=False):
    language: ReadOnly[str]
    request_id: ReadOnly[str]
    truncated: ReadOnly[bool]


def _caller_fields(payload: _PrismaTranscript) -> _CallerFields:
    language: Final[_CallerFields] = (
        {} if payload.detected_language_code is None else {"language": payload.detected_language_code}
    )
    request_id: Final[_CallerFields] = {} if payload.request_id is None else {"request_id": payload.request_id}
    truncated: Final[_CallerFields] = {} if payload.truncated is None else {"truncated": payload.truncated}
    return {**language, **request_id, **truncated}


class GnaniAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIAudioTranscriptionOptionalParams]:  # mutable-ok: the public capability API returns a list
        if _is_v25(model):
            return ["language", "languages", "keywords", "response_format"]
        return ["language", "languages", "keywords", "response_format", "prompt"]

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
        mapped: Final = {key: non_default_params.get(key) for key in ("language", "languages", "keywords", "prompt")}
        return {**optional_params, **{key: value for key, value in mapped.items() if value is not None}}

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
            raise AuthenticationError(
                message="Set GNANI_API_KEY or pass api_key for Gnani transcription", llm_provider="gnani", model=model
            )
        extra_headers: Final = as_str_mapping(optional_params.get("extra_headers")) or {}
        return {**_caller_headers({**headers, **extra_headers}), _KEY_HEADER: key}

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
        try:
            audio: Final = process_audio_file(audio_file)
        except (ValueError, OSError) as exc:
            raise BadRequestError(message="Invalid audio upload", model=model, llm_provider="gnani") from exc
        if not audio.file_content:
            raise BadRequestError(
                message="Gnani transcription requires a non-empty audio file", model=model, llm_provider="gnani"
            )
        drops_unsupported: Final = litellm_params.get("drop_params") is True or litellm.drop_params is True
        dropped: Final = _dropped_names(optional_params.get("additional_drop_params"))
        requested: Final = _without_names(
            {**(as_str_mapping(optional_params.get("extra_body")) or {}), **optional_params}, dropped
        )
        fields: Final = _without_names(
            _with_keywords_as_terms(_with_prompt_as_previous_text(requested), model, drops_unsupported), dropped
        )
        supported: Final = _V25_FIELDS if _is_v25(model) else _V3_FIELDS
        incompatible: Final = (_V25_FIELDS | _V3_FIELDS) - supported
        unsupported: Final = sorted(key for key in incompatible if fields.get(key) is not None)
        if unsupported and not drops_unsupported:
            raise UnsupportedParamsError(
                message=f"{model} does not support {', '.join(unsupported)}. To drop it, set drop_params=True.",
                model=model,
                llm_provider="gnani",
            )
        language_code: Final = _language_code(model, _language_hint(fields))
        forwarded: Final = {
            key: value for key, value in fields.items() if key not in _NOT_FORWARDED and key not in incompatible
        }
        form: Final = {key: _as_text(value) for key, value in forwarded.items() if value is not None}
        return AudioTranscriptionRequestData(
            data={**form, "model": model, "language_code": language_code},
            files={"audio_file": (Path(audio.filename).name, audio.file_content, audio.content_type)},
        )

    def transform_audio_transcription_response(self, raw_response: httpx.Response) -> TranscriptionResponse:
        try:
            payload: Final = _PrismaTranscript.model_validate_json(raw_response.content)
        except ValidationError:
            raise BaseLLMException(502, "Invalid Gnani transcription response", headers=raw_response.headers) from None
        metadata: Final = payload.model_dump(exclude={"success", "transcript"}, exclude_none=True)
        response: Final = TranscriptionResponse(text=payload.transcript, **_caller_fields(payload))
        response.hidden_params = metadata
        return response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Mapping[str, object] | httpx.Headers
    ) -> BaseLLMException:
        return BaseLLMException(status_code, error_message, headers=dict(headers))
