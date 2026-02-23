"""
Support for OVHCloud AI Endpoints `/v1/audio/transcriptions` endpoint.

Our unified API follows the OpenAI standard.
More information on our website: https://endpoints.ai.cloud.ovh.net
"""

from typing import List, Optional, Union

import httpx

from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIAudioTranscriptionOptionalParams,
)
from litellm.types.utils import FileTypes, TranscriptionResponse

from ..utils import OVHCloudException


class OVHCloudAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIAudioTranscriptionOptionalParams]:
        # OVHCloud implements the OpenAI-compatible Whisper interface.
        # We pass through the same optional params as the OpenAI Whisper API.
        return ["language", "prompt", "response_format", "timestamp_granularities", "temperature"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)
        for k, v in non_default_params.items():
            if k in supported_params:
                optional_params[k] = v
        return optional_params

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        api_base = (
            "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
            if api_base is None
            else api_base.rstrip("/")
        )
        complete_url = f"{api_base}/audio/transcriptions"
        return complete_url

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return OVHCloudException(
            message=error_message,
            status_code=status_code,
            headers=headers,
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
        if api_key is None:
            api_key = get_secret_str("OVHCLOUD_API_KEY")

        default_headers = {
            "Authorization": f"Bearer {api_key}",
            "accept": "application/json",
        }

        # Caller can override / extend headers if needed
        default_headers.update(headers or {})
        return default_headers

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
    ) -> AudioTranscriptionRequestData:
        """
        Transform the audio transcription request into OpenAI-compatible form-data.

        OVHCloud follows OpenAI's `/audio/transcriptions` format, so we:
        - Build a multipart form-data body with `file`, `model`, and optional params
        - Let the shared HTTP handler set the proper content-type boundary
        """
        processed_audio = process_audio_file(audio_file)

        # Base form fields: model + OpenAI-compatible optional params
        form_fields: dict = {
            "model": model,
        }

        # Include OpenAI-compatible optional params
        for key in self.get_supported_openai_params(model):
            value = optional_params.get(key)
            if value is not None:
                form_fields[key] = value

        files = {
            "file": (
                processed_audio.filename,
                processed_audio.file_content,
                processed_audio.content_type,
            )
        }

        return AudioTranscriptionRequestData(data=form_fields, files=files)

    def transform_audio_transcription_response(
        self,
        raw_response: httpx.Response,
    ) -> TranscriptionResponse:
        """
        Transform OVHCloud audio transcription response to OpenAI-compatible TranscriptionResponse.
        """
        try:
            response_json = raw_response.json()
        except Exception:
            raise OVHCloudException(
                message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        text = response_json.get("text") or response_json.get("transcript") or ""
        response = TranscriptionResponse(text=text)

        response._hidden_params = response_json
        return response


