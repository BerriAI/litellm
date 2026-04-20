"""
Support for Scaleway's OpenAI-compatible `/v1/audio/transcriptions` endpoint.

API reference: https://www.scaleway.com/en/developers/api/generative-apis/#path-audio-create-an-audio-transcription
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


class ScalewayAudioTranscriptionException(BaseLLMException):
    pass


class ScalewayAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIAudioTranscriptionOptionalParams]:
        return [
            "language",
            "prompt",
            "response_format",
            "temperature",
            "timestamp_granularities",
        ]

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
            "https://api.scaleway.ai/v1" if api_base is None else api_base.rstrip("/")
        )
        return f"{api_base}/audio/transcriptions"

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return ScalewayAudioTranscriptionException(
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
            api_key = get_secret_str("SCW_SECRET_KEY")

        if not api_key:
            raise ScalewayAudioTranscriptionException(
                message=(
                    "Scaleway API key not found. Pass `api_key=...` or set the "
                    "SCW_SECRET_KEY environment variable."
                ),
                status_code=401,
                headers={},
            )

        default_headers = {
            "Authorization": f"Bearer {api_key}",
            "accept": "application/json",
        }
        default_headers.update(headers or {})
        return default_headers

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
    ) -> AudioTranscriptionRequestData:
        processed_audio = process_audio_file(audio_file)

        form_fields: dict = {"model": model}
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
        content_type = (raw_response.headers.get("content-type") or "").lower()
        if "application/json" not in content_type:
            return TranscriptionResponse(text=raw_response.text)

        try:
            response_json = raw_response.json()
        except Exception:
            raise ScalewayAudioTranscriptionException(
                message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        text = response_json.get("text") or ""
        response = TranscriptionResponse(text=text)

        if "segments" in response_json:
            response["segments"] = response_json["segments"]
        if "language" in response_json:
            response["language"] = response_json["language"]

        response._hidden_params = response_json
        return response
