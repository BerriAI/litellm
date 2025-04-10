from typing import List, Optional, Union

from httpx import Headers

from litellm.llms.base_llm.audio_transcription.transformation import (
    BaseAudioTranscriptionConfig,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIAudioTranscriptionOptionalParams,
)
from litellm.types.utils import FileTypes

from ..common_utils import OpenAIError


class OpenAIWhisperAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIAudioTranscriptionOptionalParams]:
        """
        Get the supported OpenAI params for the `whisper-1` models
        """
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
        """
        Map the OpenAI params to the Whisper params
        """
        supported_params = self.get_supported_openai_params(model)
        for k, v in non_default_params.items():
            if k in supported_params:
                optional_params[k] = v
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
        api_key = api_key or get_secret_str("OPENAI_API_KEY")

        auth_header = {
            "Authorization": f"Bearer {api_key}",
        }

        headers.update(auth_header)
        return headers

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
    ) -> dict:
        """
        Transform the audio transcription request
        """

        data = {"model": model, "file": audio_file, **optional_params}

        if "response_format" not in data or (
            data["response_format"] == "text" or data["response_format"] == "json"
        ):
            data[
                "response_format"
            ] = "verbose_json"  # ensures 'duration' is received - used for cost calculation

        return data

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, Headers]
    ) -> BaseLLMException:
        return OpenAIError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
