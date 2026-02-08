"""
Sarvam AI Speech-to-Text transformation

Translates from OpenAI's `/v1/audio/transcriptions` to Sarvam's `/speech-to-text`
"""

from typing import List, Optional, Union

from httpx import Headers, Response

from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import AllMessageValues, FileTypes, TranscriptionResponse

from ...base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from ..common_utils import SARVAM_API_BASE, SarvamException


class SarvamAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    """
    Configuration for Sarvam AI Speech-to-Text

    Reference: https://docs.sarvam.ai/api-reference-docs/speech/speech-to-text

    Supported models:
    - saarika:v2.5 (default): Transcribes audio in the spoken language
    - saaras:v3: State-of-the-art model with flexible output formats

    saaras:v3 Modes (use 'mode' parameter):
    - transcribe (default): Standard transcription with formatting and number normalization
    - translate: Translates Indic speech to English
    - verbatim: Exact word-for-word without normalization
    - translit: Romanization to Latin script
    - codemix: Code-mixed text (English words in English, Indic in native script)

    Supported languages:
    hi-IN, bn-IN, kn-IN, ml-IN, mr-IN, od-IN, pa-IN, ta-IN, te-IN, en-IN, gu-IN, unknown (auto-detect)
    """

    # Valid modes for saaras:v3
    VALID_MODES = {"transcribe", "translate", "verbatim", "translit", "codemix"}

    @property
    def custom_llm_provider(self) -> str:
        return "sarvam"

    def get_supported_openai_params(self, model: str) -> List[str]:
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
                if k == "language":
                    # Map OpenAI language to Sarvam language_code
                    optional_params["language_code"] = v
                else:
                    optional_params[k] = v
        return optional_params

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, Headers]
    ) -> BaseLLMException:
        return SarvamException(
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
        Transforms the audio transcription request for Sarvam API.

        Sarvam expects multipart form data with:
        - file: audio file
        - model: model name (e.g., saarika:v2.5, saaras:v3)
        - language_code: optional language code (e.g., hi-IN, unknown for auto-detect)
        - mode: optional mode for saaras:v3 (transcribe, translate, verbatim, translit, codemix)
        """

        # Use common utility to process the audio file
        processed_audio = process_audio_file(audio_file)

        # Prepare form data
        form_data = {"model": model}

        # Map language parameter to language_code
        if "language" in optional_params:
            form_data["language_code"] = optional_params["language"]
        elif "language_code" in optional_params:
            form_data["language_code"] = optional_params["language_code"]

        # Add mode parameter (only applicable for saaras:v3)
        if "mode" in optional_params:
            form_data["mode"] = optional_params["mode"]

        # Add provider-specific parameters
        provider_specific_params = self.get_provider_specific_params(
            model=model,
            optional_params=optional_params,
            openai_params=self.get_supported_openai_params(model)
        )

        for key, value in provider_specific_params.items():
            if key not in form_data and value is not None:
                form_data[key] = str(value)

        # Prepare files
        files = {
            "file": (
                processed_audio.filename,
                processed_audio.file_content,
                processed_audio.content_type,
            )
        }

        return AudioTranscriptionRequestData(
            data=form_data,
            files=files,
        )

    def transform_audio_transcription_response(
        self,
        raw_response: Response,
    ) -> TranscriptionResponse:
        """
        Transforms the raw response from Sarvam to TranscriptionResponse format.

        Sarvam API returns:
        {
            "transcript": "...",
            "language_code": "hi-IN"
        }
        """
        response_json = raw_response.json()

        # Extract the transcript text
        text = response_json.get("transcript", "")

        # Create TranscriptionResponse object
        response = TranscriptionResponse(text=text)

        # Add metadata
        response["task"] = "transcribe"
        response["language"] = response_json.get("language_code", "unknown")

        # Store full response in hidden params
        response._hidden_params = response_json

        return response

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
            api_base = get_secret_str("SARVAM_API_BASE") or SARVAM_API_BASE

        api_base = api_base.rstrip("/")
        return f"{api_base}/speech-to-text"

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
        api_key = api_key or get_secret_str("SARVAM_API_KEY")
        if api_key is None:
            raise ValueError(
                "Sarvam API key is required. Set SARVAM_API_KEY environment variable."
            )

        headers.update({
            "api-subscription-key": api_key,
        })

        return headers
