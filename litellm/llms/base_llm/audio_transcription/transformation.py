from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIAudioTranscriptionOptionalParams,
)
from litellm.types.utils import FileTypes, ModelResponse, TranscriptionResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


@dataclass
class AudioTranscriptionRequestData:
    """
    Structured data for audio transcription requests.

    Attributes:
        data: The request data (form data for multipart, json data for regular requests)
        files: Optional files dict for multipart form data
        content_type: Optional content type override
    """

    data: Union[dict, bytes]
    files: Optional[dict] = None
    content_type: Optional[str] = None


class BaseAudioTranscriptionConfig(BaseConfig, ABC):
    @abstractmethod
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIAudioTranscriptionOptionalParams]:
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
        """
        OPTIONAL

        Get the complete url for the request

        Some providers need `model` in `api_base`
        """
        return api_base or ""

    @abstractmethod
    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
    ) -> AudioTranscriptionRequestData:
        raise NotImplementedError(
            "AudioTranscriptionConfig needs a request transformation for audio transcription models"
        )

    def transform_audio_transcription_response(
        self,
        raw_response: httpx.Response,
    ) -> TranscriptionResponse:
        raise NotImplementedError(
            "AudioTranscriptionConfig does not need a response transformation for audio transcription models"
        )

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        raise NotImplementedError(
            "AudioTranscriptionConfig does not need a request transformation for audio transcription models"
        )

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        raise NotImplementedError(
            "AudioTranscriptionConfig does not need a response transformation for audio transcription models"
        )

    def get_provider_specific_params(
        self,
        model: str,
        optional_params: dict,
        openai_params: List[OpenAIAudioTranscriptionOptionalParams],
    ) -> dict:
        """
        Get provider specific parameters that are not OpenAI compatible

        eg. if user passes `diarize=True`, we need to pass `diarize` to the provider
        but `diarize` is not an OpenAI parameter, so we need to handle it here
        """
        provider_specific_params = {}
        for key, value in optional_params.items():
            # Skip None values
            if value is None:
                continue

            # Skip excluded parameters
            if self._should_exclude_param(
                param_name=key,
                model=model,
            ):
                continue

            # Add the parameter to the provider specific params
            provider_specific_params[key] = value

        return provider_specific_params

    def _should_exclude_param(
        self,
        param_name: str,
        model: str,
    ) -> bool:
        """
        Determines if a parameter should be excluded from the query string.

        Args:
            param_name: Parameter name
            model: Model name

        Returns:
            True if the parameter should be excluded
        """
        # Parameters that are handled elsewhere or not relevant to Deepgram API
        excluded_params = {
            "model",  # Already in the URL path
            "OPENAI_TRANSCRIPTION_PARAMS",  # Internal litellm parameter
        }

        # Skip if it's an excluded parameter
        if param_name in excluded_params:
            return True

        # Skip if it's an OpenAI-specific parameter that we handle separately
        if param_name in self.get_supported_openai_params(model):
            return True

        return False
