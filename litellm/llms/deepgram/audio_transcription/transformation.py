"""
Translates from OpenAI's `/v1/audio/transcriptions` to Deepgram's `/v1/listen`
"""

import io
from typing import List, Optional, Union
from urllib.parse import urlencode

from httpx import Headers, Response

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIAudioTranscriptionOptionalParams,
)
from litellm.types.utils import FileTypes, TranscriptionResponse

from ...base_llm.audio_transcription.transformation import (
    BaseAudioTranscriptionConfig,
    LiteLLMLoggingObj,
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
    ) -> Union[dict, bytes]:
        """
        Processes the audio file input based on its type and returns the binary data.

        Args:
            audio_file: Can be a file path (str), a tuple (filename, file_content), or binary data (bytes).

        Returns:
            The binary data of the audio file.
        """
        binary_data: bytes  # Explicitly declare the type

        # Handle the audio file based on type
        if isinstance(audio_file, str):
            # If it's a file path
            with open(audio_file, "rb") as f:
                binary_data = f.read()  # `f.read()` always returns `bytes`
        elif isinstance(audio_file, tuple):
            # Handle tuple case
            _, file_content = audio_file[:2]
            if isinstance(file_content, str):
                with open(file_content, "rb") as f:
                    binary_data = f.read()  # `f.read()` always returns `bytes`
            elif isinstance(file_content, bytes):
                binary_data = file_content
            else:
                raise TypeError(
                    f"Unexpected type in tuple: {type(file_content)}. Expected str or bytes."
                )
        elif isinstance(audio_file, bytes):
            # Assume it's already binary data
            binary_data = audio_file
        elif isinstance(audio_file, io.BufferedReader) or isinstance(
            audio_file, io.BytesIO
        ):
            # Handle file-like objects
            binary_data = audio_file.read()

        else:
            raise TypeError(f"Unsupported type for audio_file: {type(audio_file)}")

        return binary_data

    def transform_audio_transcription_response(
        self,
        model: str,
        raw_response: Response,
        model_response: TranscriptionResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
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
