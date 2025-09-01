"""
Translates from OpenAI's `/v1/audio/transcriptions` to ElevenLabs's `/v1/speech-to-text`
"""

from typing import List, Optional, Union

from httpx import Headers, Response

import litellm
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
from ..common_utils import ElevenLabsException


class ElevenLabsAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    @property
    def custom_llm_provider(self) -> str:
        return litellm.LlmProviders.ELEVENLABS.value

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIAudioTranscriptionOptionalParams]:
        return ["language", "temperature"]

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
                if k == "language":
                    # Map OpenAI language format to ElevenLabs language_code
                    optional_params["language_code"] = v
                else:
                    optional_params[k] = v
        return optional_params

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, Headers]
    ) -> BaseLLMException:
        return ElevenLabsException(
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
        Transforms the audio transcription request for ElevenLabs API.
        
        Returns AudioTranscriptionRequestData with both form data and files.
        
        Returns:
            AudioTranscriptionRequestData: Structured data with form data and files
        """
        
        # Use common utility to process the audio file
        processed_audio = process_audio_file(audio_file)
        
        # Prepare form data
        form_data = {"model_id": model}

        
        #########################################################
        # Add OpenAI Compatible Parameters
        #########################################################
        for key, value in optional_params.items():
            if key in self.get_supported_openai_params(model) and value is not None:
                # Convert values to strings for form data, but skip None values
                form_data[key] = str(value)
        
        #########################################################
        # Add Provider Specific Parameters
        #########################################################
        provider_specific_params = self.get_provider_specific_params(
            model=model,
            optional_params=optional_params,
            openai_params=self.get_supported_openai_params(model)
        )

        for key, value in provider_specific_params.items():
            form_data[key] = str(value)
        #########################################################
        #########################################################
        
        # Prepare files
        files = {"file": (processed_audio.filename, processed_audio.file_content, processed_audio.content_type)}
        
        return AudioTranscriptionRequestData(
            data=form_data,
            files=files
        )


    def transform_audio_transcription_response(
        self,
        raw_response: Response,
    ) -> TranscriptionResponse:
        """
        Transforms the raw response from ElevenLabs to the TranscriptionResponse format
        """
        try:
            response_json = raw_response.json()

            # Extract the main transcript text
            text = response_json.get("text", "")

            # Create TranscriptionResponse object
            response = TranscriptionResponse(text=text)

            # Add additional metadata matching OpenAI format
            response["task"] = "transcribe"
            response["language"] = response_json.get("language_code", "unknown")
            
            # Map ElevenLabs words to OpenAI format
            if "words" in response_json:
                response["words"] = []
                for word_data in response_json["words"]:
                    # Only include actual words, skip spacing and audio events
                    if word_data.get("type") == "word":
                        response["words"].append({
                            "word": word_data.get("text", ""),
                            "start": word_data.get("start", 0),
                            "end": word_data.get("end", 0)
                        })

            # Store full response in hidden params
            response._hidden_params = response_json

            return response

        except Exception as e:
            raise ValueError(
                f"Error transforming ElevenLabs response: {str(e)}\nResponse: {raw_response.text}"
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
                get_secret_str("ELEVENLABS_API_BASE") or "https://api.elevenlabs.io"
            )
        api_base = api_base.rstrip("/")  # Remove trailing slash if present

        # ElevenLabs speech-to-text endpoint
        url = f"{api_base}/v1/speech-to-text"

        return url

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
        api_key = api_key or get_secret_str("ELEVENLABS_API_KEY")
        if api_key is None:
            raise ValueError(
                "ElevenLabs API key is required. Set ELEVENLABS_API_KEY environment variable."
            )

        auth_header = {
            "xi-api-key": api_key,
        }

        headers.update(auth_header)
        return headers 