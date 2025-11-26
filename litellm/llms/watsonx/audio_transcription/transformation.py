"""
Translates from OpenAI's `/v1/audio/transcriptions` to IBM WatsonX's `/ml/v1/audio/transcriptions`

WatsonX follows the OpenAI spec for audio transcription.
"""

from typing import List, Optional

import litellm
from litellm.types.llms.openai import OpenAIAudioTranscriptionOptionalParams

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

        URL format: {api_base}/ml/v1/audio/transcriptions?version={version}&project_id={project_id}
        """
        # Get base URL
        url = self._get_base_url(api_base=api_base)
        url = url.rstrip("/")

        # Add the audio transcription endpoint
        url = f"{url}/ml/v1/audio/transcriptions"

        # Get API params for project_id
        api_params = _get_api_params(params=optional_params.copy())

        # Add version parameter
        api_version = optional_params.pop(
            "api_version", None
        ) or litellm.WATSONX_DEFAULT_API_VERSION
        url = f"{url}?version={api_version}"

        # Add project_id parameter
        project_id = api_params.get("project_id")
        if project_id:
            url = f"{url}&project_id={project_id}"

        return url
