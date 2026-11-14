"""
Transformation logic for Hosted VLLM rerank
"""

from typing import Final

import httpx

from litellm.litellm_core_utils.url_utils import ensure_api_route
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai.transcriptions.whisper_transformation import (
    OpenAIWhisperAudioTranscriptionConfig,
)
from litellm.types.utils import FileTypes


class HostedVLLMAudioTranscriptionError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: dict | httpx.Headers | None = None,
    ):
        super().__init__(status_code=status_code, message=message, headers=headers)


class HostedVLLMAudioTranscriptionConfig(OpenAIWhisperAudioTranscriptionConfig):
    def __init__(self) -> None:
        pass

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        if api_base:
            return ensure_api_route(api_base, "v1/audio/transcriptions")
        raise ValueError("api_base must be provided for Hosted VLLM rerank")

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
    ) -> AudioTranscriptionRequestData:
        """
        Transform the audio transcription request
        """

        data: Final = {"model": model, "file": audio_file, **optional_params}

        return AudioTranscriptionRequestData(
            data=data,
        )
