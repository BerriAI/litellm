"""
Translates from OpenAI's `/v1/audio/transcriptions` to IBM WatsonX's `/ml/v1/audio/transcriptions`

WatsonX follows the OpenAI spec for audio transcription.
"""

from typing import Any, Dict, List, Optional

import litellm
from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIAudioTranscriptionOptionalParams,
)
from litellm.types.llms.watsonx import WatsonXAudioTranscriptionRequestBody
from litellm.types.utils import FileTypes

from ...base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
)
from ...openai.transcriptions.whisper_transformation import (
    OpenAIWhisperAudioTranscriptionConfig,
)
from ..common_utils import IBMWatsonXMixin, _get_api_params


class IBMWatsonXAudioTranscriptionConfig(
    IBMWatsonXMixin, OpenAIWhisperAudioTranscriptionConfig
):
    """
    IBM WatsonX Audio Transcription Config

    WatsonX follows the OpenAI spec for audio transcription, so this class
    inherits from OpenAIWhisperAudioTranscriptionConfig and uses IBMWatsonXMixin
    for authentication and URL construction.
    """

    def validate_environment(
        self,
        headers: Dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: Dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Dict:
        """
        Validate environment for audio transcription.
        
        Removes Content-Type header so httpx can set multipart/form-data automatically.
        """
        result = IBMWatsonXMixin.validate_environment(
            self,
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
        )
        # Remove Content-Type so httpx sets multipart/form-data automatically
        result.pop("Content-Type", None)
        return result

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIAudioTranscriptionOptionalParams]:
        """
        Get the supported OpenAI params for WatsonX audio transcription.
        """
        return [
            "language",
            "prompt",
            "response_format",
            "temperature",
            "timestamp_granularities",
        ]

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
    ) -> AudioTranscriptionRequestData:
        """
        Transform the audio transcription request for WatsonX.
        
        WatsonX expects multipart/form-data with:
        - file: the audio file
        - model: the model name (without watsonx/ prefix)
        - project_id: the project ID (as form field, not query param)
        - other optional params
        """
        # Use common utility to process the audio file
        processed_audio = process_audio_file(audio_file)
        
        # Get API params to extract project_id
        api_params = _get_api_params(params=optional_params.copy())
        
        # Initialize form data with required fields
        form_data: WatsonXAudioTranscriptionRequestBody = {
            "model": model,
            "project_id": api_params.get("project_id", ""),
        }
        
        # Add supported OpenAI params to form data
        supported_params = self.get_supported_openai_params(model)
        for key, value in optional_params.items():
            if key in supported_params and value is not None:
                form_data[key] = value  # type: ignore
        
        # Prepare files dict with the audio file
        files = {
            "file": (
                processed_audio.filename,
                processed_audio.file_content,
                processed_audio.content_type,
            )
        }
        
        # Convert TypedDict to regular dict for AudioTranscriptionRequestData
        form_data_dict: Dict[str, Any] = dict(form_data)
        
        return AudioTranscriptionRequestData(data=form_data_dict, files=files)

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Construct the complete URL for WatsonX audio transcription.

        URL format: {api_base}/ml/v1/audio/transcriptions?version={version}
        
        Note: project_id is sent as form data, not as a query parameter
        """
        # Get base URL
        url = self._get_base_url(api_base=api_base)
        url = url.rstrip("/")

        # Add the audio transcription endpoint
        url = f"{url}/ml/v1/audio/transcriptions"

        # Add version parameter (only version in query string, not project_id)
        api_version = optional_params.get(
            "api_version", None
        ) or litellm.WATSONX_DEFAULT_API_VERSION
        url = f"{url}?version={api_version}"

        return url
