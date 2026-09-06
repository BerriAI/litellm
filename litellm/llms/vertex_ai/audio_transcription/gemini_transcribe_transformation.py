import base64
from collections.abc import Mapping, Sequence
from typing import Final

from httpx import Headers, Response

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.audio_utils.utils import (
    normalize_transcription_language_to_bcp47,
    process_audio_file,
)
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.vertex_ai.audio_transcription.transformation import (
    SUPPORTED_RESPONSE_FORMATS,
    validate_vertex_transcription_location,
    validate_vertex_transcription_project_id,
)
from litellm.llms.vertex_ai.common_utils import VertexAIError, get_vertex_base_url
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIAudioTranscriptionOptionalParams,
)
from litellm.types.llms.vertex_ai_gemini_transcription import (
    VertexGeminiTranscriptionAudioConfig,
    VertexGeminiTranscriptionContent,
    VertexGeminiTranscriptionGenerationConfig,
    VertexGeminiTranscriptionInlineData,
    VertexGeminiTranscriptionPart,
    VertexGeminiTranscriptionRequest,
    VertexGeminiTranscriptionResponse,
)
from litellm.types.utils import (
    FileTypes,
    TranscriptionResponse,
    TranscriptionUsageInputTokenDetailsObject,
    TranscriptionUsageTokensObject,
)

DEFAULT_GEMINI_TRANSCRIBE_LOCATION: Final = "global"
AUDIO_MODALITY: Final = "AUDIO"
AMBIGUOUS_WEBM_MIME_TYPES: Final = frozenset({"application/webm", "application/x-webm", "video/webm", "video/x-webm"})


class VertexGeminiAudioTranscriptionConfig(BaseAudioTranscriptionConfig, VertexBase):
    def __init__(self) -> None:
        BaseAudioTranscriptionConfig.__init__(self)
        VertexBase.__init__(self)

    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIAudioTranscriptionOptionalParams]:  # mutable-ok: BaseAudioTranscriptionConfig signature
        return ["language", "response_format"]

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: BaseAudioTranscriptionConfig signature
        supported_params: Final = frozenset(self.get_supported_openai_params(model))
        mapped: Final = {
            **optional_params,
            **{k: v for k, v in non_default_params.items() if k in supported_params},
        }
        response_format: Final = mapped.get("response_format")
        if response_format is None or response_format in SUPPORTED_RESPONSE_FORMATS:
            return mapped
        if drop_params or litellm.drop_params:
            return {k: v for k, v in mapped.items() if k != "response_format"}
        raise UnsupportedParamsError(
            status_code=400,
            message=(
                f"Vertex AI Gemini transcription does not support response_format={response_format!r}. "
                f"Supported values: {', '.join(SUPPORTED_RESPONSE_FORMATS)}. "
                "To drop unsupported openai params from the call, set `litellm.drop_params = True`"
            ),
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | Headers,  # mutable-ok: base signature and VertexAIError take dict | Headers
    ) -> BaseLLMException:
        return VertexAIError(status_code=status_code, message=error_message, headers=headers)

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseAudioTranscriptionConfig signature
        vertex_params: Final = dict(litellm_params)
        access_token, project_id = self._ensure_access_token(
            credentials=self.safe_get_vertex_ai_credentials(vertex_params),
            project_id=self.safe_get_vertex_ai_project(vertex_params),
            custom_llm_provider="vertex_ai",
        )
        return {
            **headers,
            "Authorization": f"Bearer {access_token}",
            "x-goog-user-project": project_id,
            "Content-Type": "application/json",
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
        vertex_params: Final = dict(litellm_params)
        location: Final = validate_vertex_transcription_location(
            self.safe_get_vertex_ai_location(vertex_params), default_location=DEFAULT_GEMINI_TRANSCRIBE_LOCATION
        )
        project_id: Final = validate_vertex_transcription_project_id(
            self.safe_get_vertex_ai_project(vertex_params) or self._resolve_project_id_from_credentials(vertex_params)
        )
        base_url: Final = (api_base or get_vertex_base_url(location)).rstrip("/")
        bare_model: Final = model.removeprefix("vertex_ai/")
        model_path: Final = f"projects/{project_id}/locations/{location}/publishers/google/models/{bare_model}"
        return f"{base_url}/v1/{model_path}:generateContent"

    def _resolve_project_id_from_credentials(self, litellm_params: Mapping[str, object]) -> str:
        vertex_params: Final = dict(litellm_params)
        _, project_id = self._ensure_access_token(
            credentials=self.safe_get_vertex_ai_credentials(vertex_params),
            project_id=None,
            custom_llm_provider="vertex_ai",
        )
        return project_id

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
    ) -> AudioTranscriptionRequestData:
        processed_audio: Final = process_audio_file(audio_file)
        mime_type: Final = _audio_transcription_mime_type(audio_file, processed_audio.content_type)
        request_body: Final = VertexGeminiTranscriptionRequest(
            contents=(
                VertexGeminiTranscriptionContent(
                    role="user",
                    parts=(
                        VertexGeminiTranscriptionPart(
                            inlineData=VertexGeminiTranscriptionInlineData(
                                mimeType=mime_type,
                                data=base64.b64encode(processed_audio.file_content).decode("utf-8"),
                            )
                        ),
                    ),
                ),
            ),
            generationConfig=VertexGeminiTranscriptionGenerationConfig(
                audioTranscriptionConfig=_audio_transcription_config(optional_params.get("language"))
            ),
        )
        return AudioTranscriptionRequestData(data=dict(request_body))

    def transform_audio_transcription_response(
        self,
        raw_response: Response,
    ) -> TranscriptionResponse:
        try:
            response_json: Final = raw_response.json()
        except ValueError:
            raise VertexAIError(
                status_code=raw_response.status_code,
                message=f"Received non-JSON response from Vertex AI Gemini transcription: {raw_response.text}",
            )
        parsed: Final = VertexGeminiTranscriptionResponse.model_validate(response_json)
        texts: Final = tuple(
            part.text
            for candidate in parsed.candidates
            if candidate.content is not None
            for part in candidate.content.parts
            if part.text
        )
        response: Final = TranscriptionResponse(text=" ".join(texts))
        response["task"] = "transcribe"
        usage: Final = parsed.usageMetadata
        if usage is not None:
            audio_tokens: Final = sum(
                detail.tokenCount for detail in usage.promptTokensDetails if detail.modality == AUDIO_MODALITY
            )
            response.usage = TranscriptionUsageTokensObject(
                type="tokens",
                input_tokens=usage.promptTokenCount,
                output_tokens=usage.candidatesTokenCount,
                total_tokens=usage.totalTokenCount,
                input_token_details=TranscriptionUsageInputTokenDetailsObject(
                    audio_tokens=audio_tokens,
                    text_tokens=usage.promptTokenCount - audio_tokens,
                ),
            )
        return response


def _audio_transcription_mime_type(audio_file: FileTypes, processed_content_type: str) -> str:
    incoming_content_type: Final = audio_file[2] if isinstance(audio_file, tuple) and len(audio_file) >= 3 else None
    if isinstance(incoming_content_type, str) and incoming_content_type.split(";", 1)[0].strip().lower().startswith(
        "audio/"
    ):
        return incoming_content_type
    processed_media_type: Final = processed_content_type.split(";", 1)[0].strip().lower()
    return "audio/webm" if processed_media_type in AMBIGUOUS_WEBM_MIME_TYPES else processed_content_type


def _audio_transcription_config(language: object) -> VertexGeminiTranscriptionAudioConfig:
    if not isinstance(language, str) or not language:
        return VertexGeminiTranscriptionAudioConfig()
    return VertexGeminiTranscriptionAudioConfig(languageCodes=(normalize_transcription_language_to_bcp47(language),))
