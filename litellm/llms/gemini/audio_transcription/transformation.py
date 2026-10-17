import base64
from collections.abc import Mapping, Sequence
from typing import Final

from httpx import Headers, Response

from litellm.litellm_core_utils.audio_utils.subtitle_utils import SUBTITLE_RESPONSE_FORMATS
from litellm.litellm_core_utils.audio_utils.utils import (
    normalize_custom_vocabulary,
    normalize_gemini_transcription_mode,
    normalize_transcription_language_to_bcp47,
    process_audio_file,
)
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.gemini.common_utils import GeminiError, GeminiModelInfo
from litellm.types.llms.gemini_audio_transcription import (
    GeminiTranscriptionAudioInput,
    GeminiTranscriptionConfig,
    GeminiTranscriptionInteractionRequest,
    GeminiTranscriptionInteractionResponse,
    GeminiTranscriptionWordAnnotation,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIAudioTranscriptionOptionalParams,
)
from litellm.types.utils import (
    FileTypes,
    TranscriptionResponse,
    TranscriptionUsageInputTokenDetailsObject,
    TranscriptionUsageTokensObject,
)

INTERACTIONS_API_REVISION: Final = "2026-05-20"
WORD_INFO_ANNOTATION_TYPE: Final = "word_info"


class GeminiAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    """
    Maps OpenAI /v1/audio/transcriptions onto the Gemini Interactions API
    (POST /v1beta/interactions) for transcription models like
    gemini-3.5-transcribe. https://ai.google.dev/gemini-api/docs/transcribe

    Provider-specific params (passed as extra kwargs / form fields):
    - `mode`: "smart" (disfluency removal, self-corrections, formatting) or
      "verbatim" (Google's default). Smart mode cannot be combined with word
      timestamps or subtitle formats.
    - `custom_vocabulary`: terms to bias recognition towards, as a list or a
      comma-separated string.
    """

    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIAudioTranscriptionOptionalParams]:  # mutable-ok: BaseAudioTranscriptionConfig signature
        return ["language", "response_format", "timestamp_granularities"]

    @property
    def supports_subtitle_synthesis(self) -> bool:
        return True

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: BaseAudioTranscriptionConfig signature
        supported_params: Final = frozenset(self.get_supported_openai_params(model))
        accepted: Final = tuple((k, v) for k, v in non_default_params.items() if k in supported_params)
        return dict((*optional_params.items(), *accepted))

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | Headers,  # mutable-ok: base signature and BaseLLMException take dict | Headers
    ) -> BaseLLMException:
        return GeminiError(status_code=status_code, message=error_message, headers=headers)

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: BaseAudioTranscriptionConfig signature
        resolved_api_key: Final = GeminiModelInfo.get_api_key(api_key)
        if not resolved_api_key:
            raise GeminiError(
                status_code=401,
                message="Google API key is required. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.",
            )
        return {
            **headers,
            "Content-Type": "application/json",
            "x-goog-api-key": resolved_api_key,
            "Api-Revision": INTERACTIONS_API_REVISION,
        }

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        resolved_api_base: Final = GeminiModelInfo.get_api_base(api_base)
        return f"{resolved_api_base}/v1beta/interactions"

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
    ) -> AudioTranscriptionRequestData:
        processed_audio: Final = process_audio_file(audio_file)
        audio_input: Final = GeminiTranscriptionAudioInput(
            type="audio",
            data=base64.b64encode(processed_audio.file_content).decode("utf-8"),
            mime_type=processed_audio.content_type,
        )
        request: Final = _build_interaction_request(
            model=model,
            audio_input=audio_input,
            transcription_config=_build_transcription_config(optional_params),
        )
        return AudioTranscriptionRequestData(data=dict(request))

    def transform_audio_transcription_response(
        self,
        raw_response: Response,
    ) -> TranscriptionResponse:
        try:
            response_json: Final = raw_response.json()
        except ValueError:
            raise GeminiError(
                status_code=raw_response.status_code,
                message=f"Received non-JSON response from Gemini Interactions API: {raw_response.text}",
            )
        parsed: Final = GeminiTranscriptionInteractionResponse.model_validate(response_json)
        if parsed.status != "completed":
            raise GeminiError(
                status_code=raw_response.status_code,
                message=f"Gemini transcription interaction did not complete (status={parsed.status}): {raw_response.text}",
            )
        text_contents: Final = tuple(
            content
            for step in parsed.steps
            for content in step.content
            if content.type == "text" and content.text is not None
        )
        response: Final = TranscriptionResponse(text=" ".join(content.text or "" for content in text_contents))
        response["task"] = "transcribe"
        words: Final = tuple(
            word
            for content in text_contents
            for annotation in content.annotations
            if (word := _annotation_to_word(annotation)) is not None
        )
        if words:
            response["words"] = list(words)
            last_word_end: Final = words[-1].get("end")
            if last_word_end is not None:
                response["duration"] = last_word_end
        if parsed.usage is not None:
            audio_tokens: Final = sum(
                by_modality.tokens
                for by_modality in parsed.usage.input_tokens_by_modality
                if by_modality.modality == "audio"
            )
            response.usage = TranscriptionUsageTokensObject(
                type="tokens",
                input_tokens=parsed.usage.total_input_tokens,
                output_tokens=parsed.usage.total_output_tokens,
                total_tokens=parsed.usage.total_tokens,
                input_token_details=TranscriptionUsageInputTokenDetailsObject(
                    audio_tokens=audio_tokens,
                    text_tokens=parsed.usage.total_input_tokens - audio_tokens,
                ),
            )
        return response


_EMPTY_TRANSCRIPTION_CONFIG: Final[GeminiTranscriptionConfig] = {}
_WORD_TIMESTAMP_CONFIG: Final[GeminiTranscriptionConfig] = {
    "mode": {
        "type": "verbatim",
        "timestamp_granularities": ("word",),
        "diarization_mode": "speaker",
    },
}
_SMART_MODE_CONFIG: Final[GeminiTranscriptionConfig] = {"mode": {"type": "smart"}}
_VERBATIM_MODE_CONFIG: Final[GeminiTranscriptionConfig] = {"mode": {"type": "verbatim"}}


def _build_interaction_request(
    model: str,
    audio_input: GeminiTranscriptionAudioInput,
    transcription_config: GeminiTranscriptionConfig,
) -> GeminiTranscriptionInteractionRequest:
    if not transcription_config:
        bare_request: Final[GeminiTranscriptionInteractionRequest] = {
            "model": model.removeprefix("gemini/"),
            "input": (audio_input,),
        }
        return bare_request
    configured_request: Final[GeminiTranscriptionInteractionRequest] = {
        "model": model.removeprefix("gemini/"),
        "input": (audio_input,),
        "generation_config": {"transcription_config": transcription_config},
    }
    return configured_request


def _language_config(language: object) -> GeminiTranscriptionConfig:
    if not isinstance(language, str) or not language:
        return _EMPTY_TRANSCRIPTION_CONFIG
    language_config: Final[GeminiTranscriptionConfig] = {
        "language_codes": (normalize_transcription_language_to_bcp47(language),),
    }
    return language_config


def _wants_word_timestamps(timestamp_granularities: object, response_format: object) -> bool:
    return (isinstance(timestamp_granularities, list) and "word" in timestamp_granularities) or (
        isinstance(response_format, str) and response_format in SUBTITLE_RESPONSE_FORMATS
    )


def _mode_config(mode: object, wants_word_timestamps: bool) -> GeminiTranscriptionConfig:
    try:
        normalized_mode: Final = normalize_gemini_transcription_mode(mode)
    except (TypeError, ValueError) as error:
        raise GeminiError(status_code=400, message=str(error)) from error
    if normalized_mode == "smart":
        if wants_word_timestamps:
            raise GeminiError(
                status_code=400,
                message=(
                    "Gemini smart transcription mode cannot be combined with word timestamps "
                    "(timestamp_granularities=['word'] or a subtitle response_format)."
                ),
            )
        return _SMART_MODE_CONFIG
    if wants_word_timestamps:
        return _WORD_TIMESTAMP_CONFIG
    return _VERBATIM_MODE_CONFIG if normalized_mode == "verbatim" else _EMPTY_TRANSCRIPTION_CONFIG


def _vocabulary_config(custom_vocabulary: object) -> GeminiTranscriptionConfig:
    try:
        terms: Final = normalize_custom_vocabulary(custom_vocabulary)
    except (TypeError, ValueError) as error:
        raise GeminiError(status_code=400, message=str(error)) from error
    if not terms:
        return _EMPTY_TRANSCRIPTION_CONFIG
    vocabulary_config: Final[GeminiTranscriptionConfig] = {"custom_vocabulary": terms}
    return vocabulary_config


def _build_transcription_config(optional_params: Mapping[str, object]) -> GeminiTranscriptionConfig:
    transcription_config: Final[GeminiTranscriptionConfig] = {
        **_language_config(optional_params.get("language")),
        **_mode_config(
            optional_params.get("mode"),
            _wants_word_timestamps(
                optional_params.get("timestamp_granularities"), optional_params.get("response_format")
            ),
        ),
        **_vocabulary_config(optional_params.get("custom_vocabulary")),
    }
    return transcription_config


def _annotation_to_word(annotation: GeminiTranscriptionWordAnnotation) -> Mapping[str, str | float] | None:
    if annotation.type != WORD_INFO_ANNOTATION_TYPE or annotation.text is None:
        return None
    entries: Final = (
        ("word", annotation.text),
        ("start", _parse_offset_seconds(annotation.start_offset)),
        ("end", _parse_offset_seconds(annotation.end_offset)),
        ("speaker", annotation.speaker),
    )
    return {key: value for key, value in entries if value is not None}


def _parse_offset_seconds(offset: str | None) -> float | None:
    if offset is None or not offset.endswith("s"):
        return None
    try:
        return float(offset[:-1])
    except ValueError:
        return None
