"""
Azure AI Speech (Cognitive Services) speech-to-text transformation.

Maps OpenAI-compatible audio transcription calls to Azure Speech REST
recognition for short audio.
"""

from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlencode, urlparse

import httpx

from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIAudioTranscriptionOptionalParams,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import FileTypes, TranscriptionResponse


class AzureSpeechAudioTranscriptionException(BaseLLMException):
    pass


class AzureSpeechAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    """
    Configuration for Azure AI Speech (Cognitive Services) STT.

    Reference:
    https://learn.microsoft.com/en-us/azure/ai-services/speech-service/rest-speech-to-text-short
    """

    COGNITIVE_SERVICES_DOMAIN = "api.cognitive.microsoft.com"
    STT_SPEECH_DOMAIN = "stt.speech.microsoft.com"
    STT_ENDPOINT_PATH = "/speech/recognition/conversation/cognitiveservices/v1"
    DEFAULT_LANGUAGE = "en-US"

    def get_supported_openai_params(self, model: str) -> List[OpenAIAudioTranscriptionOptionalParams]:
        return ["language", "response_format"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model=model)
        for key, value in non_default_params.items():
            if key in supported_params:
                optional_params[key] = value
        return optional_params

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
        api_key = api_key or get_secret_str("AZURE_SPEECH_API_KEY")
        if not api_key:
            raise AzureSpeechAudioTranscriptionException(
                message="api_key is required for Azure AI Speech transcription.",
                status_code=401,
            )

        validated_headers = headers.copy()
        validated_headers["Ocp-Apim-Subscription-Key"] = api_key
        validated_headers["Content-Type"] = validated_headers.get("Content-Type", "audio/wav")
        validated_headers["Accept"] = "application/json"
        return validated_headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        api_base = api_base or get_secret_str("AZURE_SPEECH_API_BASE")
        if api_base is None:
            raise AzureSpeechAudioTranscriptionException(
                message=(
                    "api_base is required for Azure AI Speech transcription. "
                    "Use a Cognitive Services endpoint like "
                    "https://{region}.api.cognitive.microsoft.com or an STT "
                    "endpoint like https://{region}.stt.speech.microsoft.com."
                ),
                status_code=400,
            )

        base_url = self._resolve_stt_base_url(api_base=api_base)
        query_params = {
            "language": optional_params.get("language", self.DEFAULT_LANGUAGE),
            "format": self._get_azure_response_format(optional_params.get("response_format")),
        }
        return f"{base_url}{self.STT_ENDPOINT_PATH}?{urlencode(query_params)}"

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
    ) -> AudioTranscriptionRequestData:
        processed_audio = process_audio_file(audio_file)
        return AudioTranscriptionRequestData(
            data=processed_audio.file_content,
            files=None,
            content_type=processed_audio.content_type,
        )

    def transform_audio_transcription_response(
        self,
        raw_response: httpx.Response,
    ) -> TranscriptionResponse:
        response_json = raw_response.json()
        recognition_status = response_json.get("RecognitionStatus")
        if recognition_status is not None and recognition_status != "Success":
            raise AzureSpeechAudioTranscriptionException(
                message=(f"Azure AI Speech transcription failed with RecognitionStatus={recognition_status}."),
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        text = self._extract_text(response_json)
        response = TranscriptionResponse(text=text)
        response._hidden_params = response_json
        return response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return AzureSpeechAudioTranscriptionException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )

    def _resolve_stt_base_url(self, api_base: str) -> str:
        api_base = api_base.rstrip("/")
        parsed_url = urlparse(api_base)
        hostname = parsed_url.hostname or ""

        if self._is_cognitive_services_endpoint(hostname=hostname):
            region = self._extract_region_from_hostname(hostname=hostname, domain=self.COGNITIVE_SERVICES_DOMAIN)
            return self._build_stt_base_url(region=region)

        if self._is_stt_endpoint(hostname=hostname):
            return f"{parsed_url.scheme}://{hostname}"

        if self._is_azure_openai_endpoint(hostname=hostname):
            raise AzureSpeechAudioTranscriptionException(
                message=(
                    "Azure AI Speech transcription requires a Cognitive Services "
                    "or STT Speech endpoint, not an Azure OpenAI endpoint."
                ),
                status_code=400,
            )

        return api_base

    def _is_cognitive_services_endpoint(self, hostname: str) -> bool:
        return hostname == self.COGNITIVE_SERVICES_DOMAIN or hostname.endswith(f".{self.COGNITIVE_SERVICES_DOMAIN}")

    def _is_stt_endpoint(self, hostname: str) -> bool:
        return hostname == self.STT_SPEECH_DOMAIN or hostname.endswith(f".{self.STT_SPEECH_DOMAIN}")

    def _is_azure_openai_endpoint(self, hostname: str) -> bool:
        return hostname.endswith(".openai.azure.com")

    def _extract_region_from_hostname(self, hostname: str, domain: str) -> str:
        if hostname.endswith(f".{domain}"):
            return hostname[: -len(f".{domain}")]
        return ""

    def _build_stt_base_url(self, region: str) -> str:
        if region:
            return f"https://{region}.{self.STT_SPEECH_DOMAIN}"
        return f"https://{self.STT_SPEECH_DOMAIN}"

    def _get_azure_response_format(self, response_format: Optional[str]) -> str:
        if response_format == "verbose_json":
            return "detailed"
        return "simple"

    def _extract_text(self, response_json: Dict[str, Any]) -> str:
        if isinstance(response_json.get("DisplayText"), str):
            return response_json["DisplayText"]

        nbest = response_json.get("NBest")
        if isinstance(nbest, list) and nbest:
            best = nbest[0]
            if isinstance(best, dict):
                return best.get("Display") or best.get("Lexical") or ""

        return ""
