"""
Translates from OpenAI's `/v1/audio/transcriptions` to Deepgram's `/v1/listen`
"""

from typing import List, Optional, Union
from urllib.parse import urlencode

from httpx import Headers, Response

from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIAudioTranscriptionOptionalParams,
)
from litellm.types.utils import FileTypes, TranscriptionResponse

from ...base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from ..common_utils import DeepgramException


class DeepgramAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIAudioTranscriptionOptionalParams]:
        return ["language"]

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

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, Headers]
    ) -> BaseLLMException:
        return DeepgramException(
            message=error_message, status_code=status_code, headers=headers
        )

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
    ) -> AudioTranscriptionRequestData:
        """
        Processes the audio file input based on its type and returns AudioTranscriptionRequestData.
        
        For Deepgram, the binary audio data is sent directly as the request body.

        Args:
            audio_file: Can be a file path (str), a tuple (filename, file_content), or binary data (bytes).

        Returns:
            AudioTranscriptionRequestData with binary data and no files.
        """
        # Use common utility to process the audio file
        processed_audio = process_audio_file(audio_file)
        
        # Return structured data with binary content and no files
        # For Deepgram, we send binary data directly as request body
        return AudioTranscriptionRequestData(
            data=processed_audio.file_content,
            files=None
        )

    def transform_audio_transcription_response(
        self,
        raw_response: Response,
    ) -> TranscriptionResponse:
        """
        Transforms the raw response from Deepgram to the TranscriptionResponse format
        """
        try:
            response_json = raw_response.json()

            # Get the first alternative from the first channel
            first_channel = response_json["results"]["channels"][0]
            first_alternative = first_channel["alternatives"][0]

            # Extract the full transcript
            text = first_alternative["transcript"]

            # Create TranscriptionResponse object
            response = TranscriptionResponse(text=text)

            # Add additional metadata matching OpenAI format
            response["task"] = "transcribe"
            response["language"] = (
                "english"  # Deepgram auto-detects but doesn't return language
            )
            response["duration"] = response_json["metadata"]["duration"]

            # Transform words to match OpenAI format
            if "words" in first_alternative:
                response["words"] = [
                    {"word": word["word"], "start": word["start"], "end": word["end"]}
                    for word in first_alternative["words"]
                ]

            # Store full response in hidden params
            response._hidden_params = response_json

            return response

        except Exception as e:
            raise ValueError(
                f"Error transforming Deepgram response: {str(e)}\nResponse: {raw_response.text}"
            )

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        if api_base is None:
            api_base = (
                get_secret_str("DEEPGRAM_API_BASE") or "https://api.deepgram.com/v1"
            )
        api_base = api_base.rstrip("/")  # Remove trailing slash if present

        # Build query parameters including the model
        all_query_params = {"model": model}

        # Add filtered optional parameters
        additional_params = self._build_query_params(optional_params, model)
        all_query_params.update(additional_params)

        # Construct URL with proper query string encoding
        base_url = f"{api_base}/listen"
        query_string = urlencode(all_query_params)
        url = f"{base_url}?{query_string}"

        return url


    def _format_param_value(self, value) -> str:
        """
        Formats a parameter value for use in query string.

        Args:
            value: The parameter value to format

        Returns:
            Formatted string value
        """
        if isinstance(value, bool):
            return str(value).lower()
        return str(value)

    def _build_query_params(self, optional_params: dict, model: str) -> dict:
        """
        Builds a dictionary of query parameters from optional_params.

        Args:
            optional_params: Dictionary of optional parameters
            model: Model name

        Returns:
            Dictionary of filtered and formatted query parameters
        """
        query_params = {}
        provider_specific_params = self.get_provider_specific_params(
            optional_params=optional_params,
            model=model,
            openai_params=self.get_supported_openai_params(model)
        )

        for key, value in provider_specific_params.items():
            # Format and add the parameter
            formatted_value = self._format_param_value(value)
            query_params[key] = formatted_value

        return query_params

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
        api_key = api_key or get_secret_str("DEEPGRAM_API_KEY")
        return {
            "Authorization": f"Token {api_key}",
        }
