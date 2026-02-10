"""
Sarvam AI Text-to-Speech transformation

Maps OpenAI TTS spec to Sarvam TTS API
"""

import base64
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import httpx
from httpx import Headers

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)
from litellm.secret_managers.main import get_secret_str

from ..common_utils import SARVAM_API_BASE, SarvamException


if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import HttpxBinaryResponseContent
else:
    LiteLLMLoggingObj = Any
    HttpxBinaryResponseContent = Any


class SarvamTextToSpeechConfig(BaseTextToSpeechConfig):
    """
    Configuration for Sarvam AI Text-to-Speech

    Reference: https://docs.sarvam.ai/api-reference-docs/text-to-speech/convert

    Supported models:
    - bulbul:v2
    - bulbul:v3

    Available speakers:

    bulbul:v2:
      Female: Anushka (default), Manisha, Vidya, Arya
      Male: Abhilash, Karun, Hitesh

    bulbul:v3:
      Shubh (default), Prabhat, Ritu, Ashutosh, Priya, Neha, Rahul,
      Pooja, Rohan, Simran, Kavya, Amit, Dev, Ishita, Shreya,
      Ratan, Varun, Manan, Sumit, Roopa, Kabir, Aayan, Advait, Amelia, Sophia

    Supported languages:
    hi-IN, bn-IN, kn-IN, ml-IN, mr-IN, od-IN, pa-IN, ta-IN, te-IN, en-IN, gu-IN

    Pricing: ₹15 per 10K characters ($0.165 per 10K characters)
    """

    TTS_ENDPOINT_PATH = "/text-to-speech"
    DEFAULT_LANGUAGE = "en-IN"
    DEFAULT_SPEAKER_V2 = "anushka"
    DEFAULT_SPEAKER_V3 = "shubh"

    def _get_default_speaker(self, model: str) -> str:
        """Get the default speaker based on model version."""
        if "v3" in model.lower():
            return self.DEFAULT_SPEAKER_V3
        return self.DEFAULT_SPEAKER_V2

    def get_supported_openai_params(self, model: str) -> list:
        """
        Sarvam TTS supports these OpenAI parameters
        """
        return ["voice", "response_format", "speed"]

    def map_openai_params(
        self,
        model: str,
        optional_params: Dict,
        voice: Optional[Union[str, Dict]] = None,
        drop_params: bool = False,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Dict]:
        """
        Process TTS parameters for Sarvam.

        Voice must be a valid Sarvam speaker name.
        See class docstring for available speakers per model.
        """
        mapped_params: Dict[str, Any] = {}
        params = dict(optional_params) if optional_params else {}

        # Extract voice - pass through directly (no mapping)
        speaker: Optional[str] = None
        if isinstance(voice, str) and voice.strip():
            speaker = voice.strip()
        elif isinstance(voice, dict):
            for key in ("voice_id", "id", "name"):
                candidate = voice.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    speaker = candidate.strip()
                    break
        elif voice is not None:
            speaker = str(voice).strip()

        if speaker is None:
            speaker = self._get_default_speaker(model)

        # Handle speed parameter (Sarvam uses 'pace' with range 0.5-2.0)
        speed = params.pop("speed", None)
        if speed is not None:
            mapped_params["pace"] = float(speed)

        # Handle response_format (Sarvam returns base64, we decode it)
        params.pop("response_format", None)  # Not directly used by Sarvam

        # Pass through remaining params
        for key, value in params.items():
            if value is not None:
                mapped_params[key] = value

        return speaker, mapped_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Validate environment and set up authentication headers
        """
        api_key = api_key or get_secret_str("SARVAM_API_KEY")

        if api_key is None:
            raise ValueError(
                "Sarvam API key is required. Set SARVAM_API_KEY environment variable."
            )

        headers.update(
            {
                "api-subscription-key": api_key,
                "Content-Type": "application/json",
            }
        )

        return headers

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, Headers]
    ) -> BaseLLMException:
        return SarvamException(
            message=error_message, status_code=status_code, headers=headers
        )

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: Optional[str],
        optional_params: Dict,
        litellm_params: Dict,
        headers: dict,
    ) -> TextToSpeechRequestData:
        """
        Build the Sarvam TTS request payload.

        Sarvam TTS API expects:
        {
            "text": "text to synthesize",
            "target_language_code": "hi-IN",
            "speaker": "anushka",
            "model": "bulbul:v2"
        }
        """
        params = dict(optional_params) if optional_params else {}

        # Get model-appropriate default speaker
        default_speaker = self._get_default_speaker(model)

        # Build request body
        request_body: Dict[str, Any] = {
            "text": input,
            "model": model,
            "speaker": voice.lower() if voice else default_speaker,
            "target_language_code": params.pop(
                "target_language_code", self.DEFAULT_LANGUAGE
            ),
        }

        # Add pace parameter (Sarvam's speed control, range 0.5-2.0)
        if "pace" in params:
            request_body["pace"] = params.pop("pace")

        # Add any remaining provider-specific params
        for key, value in params.items():
            if value is not None:
                request_body[key] = value

        return TextToSpeechRequestData(
            dict_body=request_body,
            headers={"Content-Type": "application/json"},
        )

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> "HttpxBinaryResponseContent":
        """
        Transform Sarvam TTS response.

        Sarvam returns base64-encoded audio in JSON:
        {
            "audios": ["base64_encoded_audio_data"]
        }

        We need to decode it and wrap it in HttpxBinaryResponseContent.
        """
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        response_json = raw_response.json()
        audios = response_json.get("audios", [])

        if not audios:
            raise ValueError("Sarvam TTS response contains no audio data")

        # Decode the first audio (Sarvam returns one audio per input)
        audio_base64 = audios[0]
        audio_bytes = base64.b64decode(audio_base64)

        # Create synthetic httpx.Response with decoded audio bytes
        # (same pattern as MiniMax TTS)
        clean_headers = dict(raw_response.headers)
        clean_headers.pop("content-encoding", None)
        clean_headers.pop("transfer-encoding", None)
        clean_headers["content-length"] = str(len(audio_bytes))
        clean_headers["content-type"] = "audio/wav"

        binary_response = httpx.Response(
            status_code=200,
            headers=clean_headers,
            content=audio_bytes,
            request=raw_response.request,
        )

        return HttpxBinaryResponseContent(binary_response)

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Construct the Sarvam TTS endpoint URL.

        Note: The TTS endpoint is at https://api.sarvam.ai/text-to-speech
        (without /v1), unlike the chat completions endpoint which uses /v1.
        We ignore the api_base from providers.json if it contains /v1.
        """
        # Use explicit api_base only if it doesn't end with /v1
        if api_base and not api_base.rstrip("/").endswith("/v1"):
            base_url = api_base
        else:
            base_url = get_secret_str("SARVAM_API_BASE") or SARVAM_API_BASE
        base_url = base_url.rstrip("/")

        return f"{base_url}{self.TTS_ENDPOINT_PATH}"