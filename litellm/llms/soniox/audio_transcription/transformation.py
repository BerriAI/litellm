"""
Translates between OpenAI's `/v1/audio/transcriptions` shape and Soniox's
async transcription API (https://soniox.com/docs/stt/async/async-transcription).

This config covers parameter mapping, env validation and response shaping.
The actual orchestration (file upload -> create -> poll -> fetch -> cleanup)
lives in `litellm.llms.soniox.audio_transcription.handler`, because Soniox's
async API requires multiple HTTP calls and does not fit the single-request
contract of `base_llm_http_handler.audio_transcriptions`.
"""

from typing import Any, Dict, List, Optional, Union

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
)
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIAudioTranscriptionOptionalParams,
)
from litellm.types.utils import FileTypes, TranscriptionResponse

# Soniox-native kwargs the user can pass through `litellm.transcription(..., **kwargs)`
# in addition to the standard OpenAI params.
SONIOX_PASSTHROUGH_PARAMS: List[str] = [
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
SONIOX_HANDLER_ONLY_PARAMS: List[str] = [
    "soniox_polling_interval",
    "soniox_max_polling_attempts",
    "soniox_cleanup",
    "filename",
]


class SonioxAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    """Configuration for Soniox async speech-to-text transcription."""

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIAudioTranscriptionOptionalParams]:
        # `language` is the only OpenAI param Soniox can use directly (mapped
        # onto `language_hints`). All other Soniox features are passed through
        # as native Soniox-named kwargs.
        return ["language"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        # Translate the OpenAI `language` param into Soniox `language_hints`.
        if "language" in non_default_params and non_default_params["language"]:
            language = non_default_params["language"]
            existing_hints = optional_params.get("language_hints")
            if not existing_hints:
                optional_params["language_hints"] = [language]
            elif language not in existing_hints:
                optional_params["language_hints"] = [language] + list(existing_hints)

        # Pass through Soniox-native kwargs unchanged.
        for key in SONIOX_PASSTHROUGH_PARAMS + SONIOX_HANDLER_ONLY_PARAMS:
            if key in non_default_params and non_default_params[key] is not None:
                optional_params[key] = non_default_params[key]

        return optional_params

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, Headers]
    ) -> BaseLLMException:
        return SonioxException(
            message=error_message, status_code=status_code, headers=headers
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        resolved_key = get_soniox_api_key(api_key)
        if not resolved_key:
            raise SonioxException(
                message=(
                    "Missing Soniox API key. Set the SONIOX_API_KEY environment "
                    "variable or pass api_key=... to litellm.transcription()."
                ),
                status_code=401,
                headers=None,
            )

        merged_headers: Dict[str, str] = {
            "Authorization": f"Bearer {resolved_key}",
        }
        if headers:
            merged_headers.update(headers)
        return merged_headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
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
        body: Dict[str, Any] = {"model": model}

        for key in SONIOX_PASSTHROUGH_PARAMS:
            value = optional_params.get(key)
            if value is not None:
                body[key] = value

        return AudioTranscriptionRequestData(
            data=body, files=None, content_type="application/json"
        )

    def transform_audio_transcription_response(
        self,
        raw_response: Response,
    ) -> TranscriptionResponse:
        """
        Build a TranscriptionResponse from a Soniox transcript payload.

        `raw_response.json()` may be either:
          - a Soniox transcript object: `{"id": "...", "text": "...", "tokens": [...]}`
          - or a merged envelope: `{"transcription": {...}, "transcript": {...}}`
            produced by the handler so transcription metadata is also available.
        """
        try:
            payload = raw_response.json()
        except Exception as exc:
            raise SonioxException(
                message=f"Failed to parse Soniox response: {exc}",
                status_code=getattr(raw_response, "status_code", 500),
                headers=getattr(raw_response, "headers", None),
            )

        return self._build_response_from_payload(payload)

    def _build_response_from_payload(
        self, payload: Dict[str, Any]
    ) -> TranscriptionResponse:
        """Shared response-building logic (also used by the handler)."""
        transcription_meta: Dict[str, Any] = {}
        transcript: Dict[str, Any]

        if isinstance(payload, dict) and "transcript" in payload:
            transcription_meta = payload.get("transcription") or {}
            transcript = payload.get("transcript") or {}
        else:
            transcript = payload if isinstance(payload, dict) else {}

        tokens: List[Dict[str, Any]] = transcript.get("tokens") or []

        # Decide what to put in `text`:
        #   - If diarization or language ID was used (i.e. tokens carry speaker
        #     or language info), render with the SDK-style tagging.
        #   - Otherwise prefer the API-provided plain `text` field, falling back
        #     to a simple concat of token texts.
        has_speaker = any(t.get("speaker") is not None for t in tokens)
        has_language = any(t.get("language") is not None for t in tokens)

        if (has_speaker or has_language) and tokens:
            text = render_soniox_tokens(tokens)
        elif transcript.get("text"):
            text = transcript["text"]
        elif tokens:
            text = "".join(t.get("text", "") for t in tokens)
        else:
            text = ""

        response = TranscriptionResponse(text=text)
        response["task"] = "transcribe"

        # Best-effort metadata fields matching OpenAI's verbose_json shape.
        if transcription_meta.get("audio_duration_ms") is not None:
            try:
                response["duration"] = (
                    float(transcription_meta["audio_duration_ms"]) / 1000.0
                )
            except (TypeError, ValueError):
                pass

        # Surface a representative language if all tokens agree.
        if has_language:
            languages = {t.get("language") for t in tokens if t.get("language")}
            if len(languages) == 1:
                response["language"] = next(iter(languages))

        # Stash the raw Soniox payload so power-users can read tokens, segments,
        # speaker/language data, etc.
        response._hidden_params = {
            "soniox_raw": {
                "transcription": transcription_meta,
                "transcript": transcript,
            }
        }
        return response
