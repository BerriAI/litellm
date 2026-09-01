"""
SLNG Audio Transcription transformation

Translates from OpenAI's `/v1/audio/transcriptions` to SLNG's `/v1/stt/` endpoint
Reference: https://docs.slng.ai/api-reference/stt
"""

from typing import Final
from urllib.parse import urljoin

from httpx import Headers, Response

from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import OpenAIAudioTranscriptionOptionalParams
from litellm.types.utils import FileTypes, TranscriptionResponse

from ...base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from ..common_utils import SlngException, get_slng_api_base, get_slng_api_key


class SlngAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    """
    Configuration for SLNG Speech-to-Text (Audio Transcription)

    SLNG provides unified access to 8 STT models from providers including
    Deepgram, Speechmatics, Sarvam, Soniox, and others across regional infrastructure.

    Reference: https://docs.slng.ai/api-reference/stt
    """

    def get_supported_openai_params(self, model: str) -> list[OpenAIAudioTranscriptionOptionalParams]:
        """
        SLNG STT supports these OpenAI transcription parameters
        """
        return ["language", "prompt", "response_format", "temperature", "timestamp_granularities"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI transcription parameters to SLNG format
        """
        supported_params: Final = self.get_supported_openai_params(model)
        for k, v in non_default_params.items():
            if k in supported_params:
                optional_params[k] = v
        return optional_params

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | Headers
    ) -> BaseLLMException:
        """
        Return SLNG-specific exception for transcription errors
        """
        return SlngException(
            message=error_message,
            status_code=status_code,
            headers=headers
        )

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
    ) -> AudioTranscriptionRequestData:
        """
        Process the audio file and prepare the request data for SLNG STT API

        Args:
            model: Model identifier (e.g., "deepgram/nova:3-en")
            audio_file: Can be file path (str), tuple (filename, content), binary data (bytes), or URL string
            optional_params: Additional parameters like language, punctuate, etc.
            litellm_params: LiteLLM internal params

        Returns:
            AudioTranscriptionRequestData with multipart form data
        """
        # Process the audio file using common utility
        processed_audio: Final = process_audio_file(audio_file)

        # SLNG API accepts either binary audio upload or URL-based requests
        # For binary uploads, use multipart/form-data with 'audio' field
        # For URL-based, use JSON body with 'url' field

        # Check if this is a URL-based request
        if isinstance(audio_file, str) and (audio_file.startswith('http://') or audio_file.startswith('https://')):
            # URL-based transcription
            form_data = {
                "url": audio_file,
            }

            # Add language if specified
            language = optional_params.get("language")
            if language:
                form_data["language"] = language

            # Add other SLNG-specific params
            for key in ["punctuate", "smart_format", "numerals", "profanity_filter",
                       "keywords", "utterances", "paragraphs", "diarize"]:
                if key in optional_params:
                    form_data[key] = optional_params[key]

            return AudioTranscriptionRequestData(
                data=form_data,
                files=None,
                content_type="application/json"
            )

        else:
            # Binary audio upload
            files = {
                "audio": (
                    processed_audio.filename or "audio.wav",
                    processed_audio.file_content,
                    processed_audio.mime_type or "audio/wav"
                )
            }

            # Form data with additional parameters
            form_data = {}

            # Add language if specified
            language = optional_params.get("language")
            if language:
                form_data["language"] = language

            # Add other SLNG-specific params
            for key in ["punctuate", "smart_format", "numerals", "profanity_filter",
                       "keywords", "utterances", "paragraphs", "diarize"]:
                if key in optional_params:
                    form_data[key] = optional_params[key]

            return AudioTranscriptionRequestData(
                data=form_data if form_data else {},
                files=files,
                content_type="multipart/form-data"
            )

    def transform_audio_transcription_response(
        self,
        raw_response: Response,
    ) -> TranscriptionResponse:
        """
        Transform the raw SLNG STT response to TranscriptionResponse format

        SLNG returns a structure like:
        {
          "results": {
            "channels": [{
              "alternatives": [{
                "transcript": "string",
                "confidence": 0.98
              }],
              "detected_language": "string"
            }]
          },
          "metadata": {
            "request_id": "string",
            "model": "nova-3"
          }
        }
        """
        try:
            response_json: Final = raw_response.json()

            # Extract transcript from SLNG response structure
            first_channel: Final = response_json["results"]["channels"][0]
            first_alternative: Final = first_channel["alternatives"][0]

            # Get the transcript text
            text = first_alternative["transcript"]

            # Create TranscriptionResponse object
            response: Final = TranscriptionResponse(text=text)

            # Add OpenAI-compatible metadata
            response["task"] = "transcribe"

            # Add detected language
            detected_language: Final = first_channel.get("detected_language")
            response["language"] = detected_language if detected_language else "en"

            # Add duration if available in metadata
            if "duration" in response_json.get("metadata", {}):
                response["duration"] = response_json["metadata"]["duration"]

            # Transform words to match OpenAI format if available
            if "words" in first_alternative:
                response["words"] = [
                    {
                        "word": word.get("word", word.get("text", "")),
                        "start": word.get("start", 0.0),
                        "end": word.get("end", 0.0)
                    }
                    for word in first_alternative["words"]
                ]

            # Store full SLNG response in hidden params
            response._hidden_params = response_json

            return response

        except Exception as e:
            raise ValueError(
                f"Error transforming SLNG transcription response: {e}\nResponse: {raw_response.text}"
            )

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        """
        Construct the complete SLNG STT endpoint URL

        Args:
            api_base: Base API URL
            api_key: API key (for potential use in URL construction)
            model: Model identifier (e.g., "deepgram/nova:3-en")
            optional_params: Additional parameters
            litellm_params: LiteLLM internal params
            stream: Whether this is a streaming request

        Returns:
            Complete API endpoint URL
        """
        resolved_api_base = get_slng_api_base(api_base)

        # SLNG STT endpoint structure: POST /v1/stt/slng/{provider}/{model_variant}
        # Example: /v1/stt/slng/deepgram/nova:3-en
        endpoint_path = f"/v1/stt/slng/{model}"

        return urljoin(resolved_api_base, endpoint_path)
