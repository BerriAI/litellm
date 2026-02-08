"""
Translates from OpenAI's `/v1/audio/transcriptions` to Mistral's `/v1/audio/transcriptions`

Mistral's transcription API is largely OpenAI-compatible but adds:
- `diarize`: bool - enable speaker diarization
- `context_bias`: str - comma-separated terms to bias the model
- `file_url`: str - URL to an audio file (alternative to file upload)
- `timestamp_granularities`: list - supports "segment" and "word"

Reference: https://docs.mistral.ai/capabilities/audio_transcription/
"""

from typing import List, Optional, Union

from httpx import Headers, Response

from litellm._logging import verbose_logger
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


class MistralAudioTranscriptionError(BaseLLMException):
    pass


class MistralAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    """
    Configuration for Mistral audio transcription via the Voxtral models.

    Mistral's POST /v1/audio/transcriptions accepts:
      - file: audio file (multipart upload)
      - model: e.g. "voxtral-mini-latest" or "voxtral-mini-2602"
      - language: ISO language code (optional)
      - response_format: "json", "text", "verbose_json" (optional)
      - timestamp_granularities: ["segment"] or ["word"] (optional)
      - diarize: bool (optional, Mistral-specific)
      - context_bias: str (optional, Mistral-specific)

    Reference: https://docs.mistral.ai/capabilities/audio_transcription/
    """

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIAudioTranscriptionOptionalParams]:
        """
        Return the OpenAI-compatible params that Mistral supports.
        """
        return ["language", "response_format", "timestamp_granularities"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI-compatible params through to optional_params.
        """
        supported_params = self.get_supported_openai_params(model)
        for k, v in non_default_params.items():
            if k in supported_params:
                optional_params[k] = v
        return optional_params

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, Headers]
    ) -> BaseLLMException:
        return MistralAudioTranscriptionError(
            message=error_message, status_code=status_code, headers=headers
        )

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
        """
        Validate environment and return auth headers for Mistral.
        """
        api_key = api_key or get_secret_str("MISTRAL_API_KEY")

        if api_key is None:
            raise ValueError(
                "Missing Mistral API Key - set MISTRAL_API_KEY in environment "
                "or pass api_key in litellm_params"
            )

        return {
            "Authorization": f"Bearer {api_key}",
        }

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
        Build the Mistral transcription endpoint URL.

        Default: https://api.mistral.ai/v1/audio/transcriptions
        """
        if api_base is None:
            api_base = (
                get_secret_str("MISTRAL_API_BASE")
                or "https://api.mistral.ai/v1"
            )

        api_base = api_base.rstrip("/")
        return f"{api_base}/audio/transcriptions"

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
    ) -> AudioTranscriptionRequestData:
        """
        Build multipart form data for Mistral's /v1/audio/transcriptions.

        Mistral accepts a standard multipart form similar to OpenAI Whisper,
        plus Mistral-specific fields (diarize, context_bias).
        """
        # Process the audio file into a consistent format
        processed_audio = process_audio_file(audio_file)

        # Build multipart file payload
        file_name = processed_audio.filename or "audio.mp3"
        files = {
            "file": (file_name, processed_audio.file_content),
        }

        # Build form data
        data: dict = {
            "model": model,
        }

        # Standard OpenAI-compatible params
        if "language" in optional_params:
            data["language"] = optional_params["language"]

        if "response_format" in optional_params:
            data["response_format"] = optional_params["response_format"]

        if "timestamp_granularities" in optional_params:
            granularities = optional_params["timestamp_granularities"]
            if isinstance(granularities, list):
                for g in granularities:
                    # multipart forms need repeated keys for arrays
                    data.setdefault("timestamp_granularities[]", [])
                    data["timestamp_granularities[]"].append(g)
            else:
                data["timestamp_granularities[]"] = granularities

        # Mistral-specific params (passed through via provider_specific_params / extra kwargs)
        provider_specific = self.get_provider_specific_params(
            model=model,
            optional_params=optional_params,
            openai_params=self.get_supported_openai_params(model),
        )

        if "diarize" in provider_specific:
            data["diarize"] = str(provider_specific["diarize"]).lower()

        if "context_bias" in provider_specific:
            data["context_bias"] = provider_specific["context_bias"]

        verbose_logger.debug(
            "Mistral audio transcription request - model: %s, file: %s, params: %s",
            model,
            file_name,
            {k: v for k, v in data.items() if k != "model"},
        )

        return AudioTranscriptionRequestData(data=data, files=files)

    def transform_audio_transcription_response(
        self,
        raw_response: Response,
    ) -> TranscriptionResponse:
        """
        Transform Mistral's transcription response to LiteLLM's TranscriptionResponse.

        Mistral returns an OpenAI-compatible response with:
        - text: the transcribed text
        - optional: segments, words, duration, language
        - when diarize=true: segments include speaker labels
        """
        try:
            response_json = raw_response.json()
        except Exception as e:
            raise ValueError(
                f"Error parsing Mistral transcription response: {e}\n"
                f"Raw response: {raw_response.text}"
            )

        text = response_json.get("text", "")
        response = TranscriptionResponse(text=text)

        # Standard fields
        response["task"] = "transcribe"

        if "language" in response_json:
            response["language"] = response_json["language"]

        if "duration" in response_json:
            response["duration"] = response_json["duration"]

        # Segments (may include speaker info when diarize=true)
        if "segments" in response_json:
            response["segments"] = response_json["segments"]

        # Word-level timestamps
        if "words" in response_json:
            response["words"] = response_json["words"]

        # Preserve the full response for consumers who need Mistral-specific data
        response._hidden_params = response_json

        return response
