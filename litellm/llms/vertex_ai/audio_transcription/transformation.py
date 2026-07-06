import base64

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
from litellm.llms.vertex_ai.common_utils import VertexAIError, validate_vertex_location
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIAudioTranscriptionOptionalParams,
)
from litellm.types.llms.vertex_ai_speech_to_text import (
    VertexSpeechToTextAutoDecodingConfig,
    VertexSpeechToTextRecognitionConfig,
    VertexSpeechToTextRecognitionFeatures,
    VertexSpeechToTextRecognizeRequest,
    VertexSpeechToTextRecognizeResponse,
)
from litellm.types.utils import FileTypes, TranscriptionResponse

DEFAULT_SPEECH_TO_TEXT_LOCATION = "us"
AUTO_LANGUAGE_CODE = "auto"
SUPPORTED_RESPONSE_FORMATS = ("json", "text")
_URL_UNSAFE_PROJECT_CHARS = ("/", "?", "#", "\\", ":", " ", "\t", "\n", "\r")


class VertexAIAudioTranscriptionConfig(BaseAudioTranscriptionConfig, VertexBase):
    def __init__(self) -> None:
        BaseAudioTranscriptionConfig.__init__(self)
        VertexBase.__init__(self)

    def get_supported_openai_params(self, model: str) -> list[OpenAIAudioTranscriptionOptionalParams]:
        return ["language", "response_format"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)
        mapped = {
            **optional_params,
            **{k: v for k, v in non_default_params.items() if k in supported_params},
        }
        response_format = mapped.get("response_format")
        if response_format is None or response_format in SUPPORTED_RESPONSE_FORMATS:
            return mapped
        if drop_params or litellm.drop_params:
            return {k: v for k, v in mapped.items() if k != "response_format"}
        raise UnsupportedParamsError(
            status_code=400,
            message=(
                f"Google Speech-to-Text does not support response_format={response_format!r}. "
                f"Supported values: {', '.join(SUPPORTED_RESPONSE_FORMATS)}. "
                "To drop unsupported openai params from the call, set `litellm.drop_params = True`"
            ),
        )

    def get_error_class(self, error_message: str, status_code: int, headers: dict | Headers) -> BaseLLMException:
        return VertexAIError(status_code=status_code, message=error_message, headers=headers)

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
        access_token, project_id = self._ensure_access_token(
            credentials=self.safe_get_vertex_ai_credentials(litellm_params),
            project_id=self.safe_get_vertex_ai_project(litellm_params),
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
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        location = self._validate_location(self.safe_get_vertex_ai_location(litellm_params))
        project_id = self._validate_project_id(
            self.safe_get_vertex_ai_project(litellm_params) or self._resolve_project_id_from_credentials(litellm_params)
        )
        host = "speech.googleapis.com" if location == "global" else f"{location}-speech.googleapis.com"
        base_url = (api_base or f"https://{host}").rstrip("/")
        return f"{base_url}/v2/projects/{project_id}/locations/{location}/recognizers/_:recognize"

    @staticmethod
    def _validate_location(location: str | None) -> str:
        try:
            return validate_vertex_location(location or DEFAULT_SPEECH_TO_TEXT_LOCATION)
        except ValueError as e:
            raise VertexAIError(status_code=400, message=str(e)) from e

    @staticmethod
    def _validate_project_id(project_id: str) -> str:
        if not project_id or ".." in project_id or any(c in project_id for c in _URL_UNSAFE_PROJECT_CHARS):
            raise VertexAIError(status_code=400, message=f"Invalid vertex_project format: {project_id!r}")
        return project_id

    def _resolve_project_id_from_credentials(self, litellm_params: dict) -> str:
        _, project_id = self._ensure_access_token(
            credentials=self.safe_get_vertex_ai_credentials(litellm_params),
            project_id=None,
            custom_llm_provider="vertex_ai",
        )
        return project_id

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
    ) -> AudioTranscriptionRequestData:
        processed_audio = process_audio_file(audio_file)
        language = optional_params.get("language")
        language_codes = (
            [normalize_transcription_language_to_bcp47(language)]
            if isinstance(language, str) and language
            else [AUTO_LANGUAGE_CODE]
        )
        request_body = VertexSpeechToTextRecognizeRequest(
            config=VertexSpeechToTextRecognitionConfig(
                model=model.removeprefix("vertex_ai/"),
                languageCodes=language_codes,
                features=VertexSpeechToTextRecognitionFeatures(enableAutomaticPunctuation=True),
                autoDecodingConfig=VertexSpeechToTextAutoDecodingConfig(),
            ),
            content=base64.b64encode(processed_audio.file_content).decode("utf-8"),
        )
        return AudioTranscriptionRequestData(data=dict(request_body))

    def transform_audio_transcription_response(
        self,
        raw_response: Response,
    ) -> TranscriptionResponse:
        try:
            response_json = raw_response.json()
        except ValueError:
            raise VertexAIError(
                status_code=raw_response.status_code,
                message=f"Received non-JSON response from Google Speech-to-Text: {raw_response.text}",
            )
        parsed = VertexSpeechToTextRecognizeResponse.model_validate(response_json)
        transcripts = tuple(
            result.alternatives[0].transcript
            for result in parsed.results
            if result.alternatives and result.alternatives[0].transcript
        )
        response = TranscriptionResponse(text=" ".join(transcripts))
        response["task"] = "transcribe"
        detected_language = next((result.languageCode for result in parsed.results if result.languageCode), None)
        if detected_language is not None:
            response["language"] = detected_language
        billed_duration = _parse_duration_seconds(parsed.metadata.totalBilledDuration if parsed.metadata else None)
        if billed_duration is not None:
            response["duration"] = billed_duration
        response._hidden_params = response_json
        return response


def _parse_duration_seconds(duration: str | None) -> float | None:
    if duration is None or not duration.endswith("s"):
        return None
    try:
        return float(duration[:-1])
    except ValueError:
        return None
