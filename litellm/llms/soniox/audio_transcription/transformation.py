"""
Translates between OpenAI's `/v1/audio/transcriptions` shape and Soniox's
async transcription API (https://soniox.com/docs/stt/async/async-transcription).

This config covers parameter mapping, env validation and response shaping.
The actual orchestration (file upload -> create -> poll -> fetch -> cleanup)
lives in `litellm.llms.soniox.audio_transcription.handler`, because Soniox's
async API requires multiple HTTP calls and does not fit the single-request
contract of `base_llm_http_handler.audio_transcriptions`.
"""

from typing import Any, Final

from httpx import Headers, Response

from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.soniox.common_utils import (
    SonioxException,
    get_soniox_api_base,
    get_soniox_api_key,
    render_soniox_tokens,
    render_soniox_tokens_as_srt,
    render_soniox_tokens_as_vtt,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIAudioTranscriptionOptionalParams,
)
from litellm.types.utils import FileTypes, TranscriptionResponse

# Soniox-native kwargs the user can pass through `litellm.transcription(..., **kwargs)`
# in addition to the standard OpenAI params.
SONIOX_PASSTHROUGH_PARAMS: Final[list[str]] = [
    "language_hints",
    "language_hints_strict",
    "enable_language_identification",
    "enable_speaker_diarization",
    "context",
    "translation",
    "client_reference_id",
    "webhook_url",
    "webhook_auth_header_name",
    "webhook_auth_header_value",
    "audio_url",
    "file_id",
]

# Handler-only kwargs (consumed by the handler, not sent to Soniox).
SONIOX_HANDLER_ONLY_PARAMS: Final[list[str]] = [
    "soniox_polling_interval",
    "soniox_max_polling_attempts",
    "soniox_cleanup",
    "filename",
]


class SonioxAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    """Configuration for Soniox async speech-to-text transcription."""

    def get_supported_openai_params(self, model: str) -> list[OpenAIAudioTranscriptionOptionalParams]:
        # `language` is mapped onto Soniox's `language_hints`.
        # `response_format` is handled by LiteLLM (Soniox doesn't support
        # SRT/VTT natively but we synthesize them from token timestamps).
        return ["language", "response_format"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        # Translate the OpenAI `language` param into Soniox `language_hints`.
        if "language" in non_default_params and non_default_params["language"]:
            language: Final = non_default_params["language"]
            existing_hints: Final = optional_params.get("language_hints")
            if not existing_hints:
                optional_params["language_hints"] = [language]
            elif language not in existing_hints:
                optional_params["language_hints"] = [language] + list(existing_hints)

        # Capture response_format for post-processing (not sent to Soniox API).
        if "response_format" in non_default_params:
            optional_params["response_format"] = non_default_params["response_format"]

        # Pass through Soniox-native kwargs unchanged.
        for key in SONIOX_PASSTHROUGH_PARAMS + SONIOX_HANDLER_ONLY_PARAMS:
            if key in non_default_params and non_default_params[key] is not None:
                optional_params[key] = non_default_params[key]

        return optional_params

    def get_error_class(self, error_message: str, status_code: int, headers: dict | Headers) -> BaseLLMException:
        return SonioxException(message=error_message, status_code=status_code, headers=headers)

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        resolved_key: Final = get_soniox_api_key(api_key)
        if not resolved_key:
            raise SonioxException(
                message=(
                    "Missing Soniox API key. Set the SONIOX_API_KEY environment "
                    "variable or pass api_key=... to litellm.transcription()."
                ),
                status_code=401,
                headers=None,
            )

        merged_headers: Final[dict[str, str]] = {
            "Authorization": f"Bearer {resolved_key}",
        }
        if headers:
            merged_headers.update(headers)
        return merged_headers

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        # The handler builds per-call URLs (uploads, create, poll, fetch, delete);
        # we just return the resolved base.
        return get_soniox_api_base(api_base)

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
    ) -> AudioTranscriptionRequestData:
        """
        Build the JSON body for `POST /v1/transcriptions`.

        The handler is responsible for the file upload (if `audio_file` is bytes)
        and for filling in `file_id`/`audio_url`. This method exists so the
        config can be exercised in isolation by unit tests.
        """
        body: Final[dict[str, Any]] = {"model": model}

        for key in SONIOX_PASSTHROUGH_PARAMS:
            value = optional_params.get(key)
            if value is not None:
                body[key] = value

        return AudioTranscriptionRequestData(data=body, files=None, content_type="application/json")

    def transform_audio_transcription_response(
        self,
        raw_response: Response,
        model_response: TranscriptionResponse | None = None,
    ) -> TranscriptionResponse:
        """
        Build a TranscriptionResponse from a Soniox transcript payload.

        `raw_response.json()` may be either:
          - a Soniox transcript object: `{"id": "...", "text": "...", "tokens": [...]}`
          - or a merged envelope: `{"transcription": {...}, "transcript": {...}}`
            produced by the handler so transcription metadata is also available.
        """
        try:
            payload: Final = raw_response.json()
        except Exception as exc:
            raise SonioxException(
                message=f"Failed to parse Soniox response: {exc}",
                status_code=getattr(raw_response, "status_code", 500),
                headers=getattr(raw_response, "headers", None),
            )

        return self._build_response_from_payload(payload, model_response=model_response)

    def _build_response_from_payload(
        self,
        payload: dict[str, Any],
        model_response: TranscriptionResponse | None = None,
        response_format: str | None = None,
    ) -> TranscriptionResponse:
        """Shared response-building logic (also used by the handler)."""
        transcription_meta: dict[str, Any] = {}
        transcript: dict[str, Any]

        if isinstance(payload, dict) and "transcript" in payload:
            transcription_meta = payload.get("transcription") or {}
            transcript = payload.get("transcript") or {}
        else:
            transcript = payload if isinstance(payload, dict) else {}

        tokens: Final[list[dict[str, Any]]] = transcript.get("tokens") or []

        # Decide what to put in `text` based on response_format:
        #   - "srt": render tokens as SRT subtitles (synthesized from timestamps)
        #   - "vtt": render tokens as WebVTT subtitles (synthesized from timestamps)
        #   - "verbose_json": return JSON with word-level timing (handled below)
        #   - "text" / "json" / None: default plain text rendering
        if response_format == "srt" and tokens:
            text = render_soniox_tokens_as_srt(tokens)
        elif response_format == "vtt" and tokens:
            text = render_soniox_tokens_as_vtt(tokens)
        else:
            # Default text rendering (also used for "json", "text",
            # "verbose_json")
            has_speaker: Final = any(t.get("speaker") is not None for t in tokens)
            has_language = any(t.get("language") is not None for t in tokens)

            if (has_speaker or has_language) and tokens:
                text = render_soniox_tokens(tokens)
            elif transcript.get("text"):
                text = transcript["text"]
            elif tokens:
                text = "".join(t.get("text", "") for t in tokens)
            else:
                text = ""

        response: Final = model_response or TranscriptionResponse(text=text)
        response.text = text
        response["task"] = "transcribe"

        # Best-effort metadata fields matching OpenAI's verbose_json shape.
        if transcription_meta.get("audio_duration_ms") is not None:
            try:
                response["duration"] = float(transcription_meta["audio_duration_ms"]) / 1000.0
            except (TypeError, ValueError):
                pass

        # Surface a representative language if all tokens agree.
        has_language = any(t.get("language") is not None for t in tokens)
        if has_language:
            languages: Final = {t.get("language") for t in tokens if t.get("language")}
            if len(languages) == 1:
                response["language"] = next(iter(languages))

        # For verbose_json, include word-level timing from tokens.
        if response_format == "verbose_json" and tokens:
            words: Final[list[dict[str, Any]]] = []
            for token in tokens:
                word_entry: dict[str, Any] = {"word": token.get("text", "")}
                if token.get("start_ms") is not None:
                    word_entry["start"] = float(token["start_ms"]) / 1000.0
                if token.get("end_ms") is not None:
                    word_entry["end"] = float(token["end_ms"]) / 1000.0
                words.append(word_entry)
            if words:
                response["words"] = words

        # Stash the raw Soniox payload so power-users can read tokens, segments,
        # speaker/language data, etc.
        response._hidden_params.update(
            {
                "soniox_raw": {
                    "transcription": transcription_meta,
                    "transcript": transcript,
                }
            }
        )
        return response
