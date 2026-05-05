"""
Transformation logic for Hosted VLLM rerank
"""

from typing import Optional, Union

import httpx

from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
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
        headers: Optional[Union[dict, httpx.Headers]] = None,
    ):
        super().__init__(status_code=status_code, message=message, headers=headers)


class HostedVLLMAudioTranscriptionConfig(OpenAIWhisperAudioTranscriptionConfig):
    def __init__(self) -> None:
        pass

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        if api_base:
            # Remove trailing slashes and ensure clean base URL
            api_base = api_base.rstrip("/")
            if not api_base.endswith("/v1/audio/transcriptions"):
                api_base = f"{api_base}/v1/audio/transcriptions"
            return api_base
        raise ValueError("api_base must be provided for Hosted VLLM rerank")

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
    ) -> AudioTranscriptionRequestData:
        """
        Transform the audio transcription request into multipart form-data.

        vLLM speaks the OpenAI transcription protocol but does not support
        verbose_json output, so we skip the parent's verbose_json override and
        pass through whatever response_format the caller specified.
        Filtering / coercion of optional_params follows the same rules as the
        parent (only supported params; bools → str; lists → comma-joined).
        """
        data: dict = {"model": model}
        for key in self.get_supported_openai_params(model):
            value = optional_params.get(key)
            if value is None:
                continue
            if isinstance(value, bool):
                data[key] = "true" if value else "false"
            elif isinstance(value, (list, tuple)):
                data[key] = ",".join(str(v) for v in value)
            else:
                data[key] = value

        processed = process_audio_file(audio_file)
        files = {
            "file": (processed.filename, processed.file_content, processed.content_type)
        }

        return AudioTranscriptionRequestData(data=data, files=files)
