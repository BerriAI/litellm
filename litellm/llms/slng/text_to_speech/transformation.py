"""
SLNG Text-to-Speech transformation

Maps OpenAI TTS spec to SLNG TTS API
Reference: https://docs.slng.ai/api-reference/tts
"""

from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

import httpx

import litellm
from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)
from litellm.secret_managers.main import get_secret_str

from ..common_utils import SlngException, get_slng_api_base, get_slng_api_key

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import HttpxBinaryResponseContent
else:
    LiteLLMLoggingObj = Any
    HttpxBinaryResponseContent = Any


class SlngTextToSpeechConfig(BaseTextToSpeechConfig):
    """
    Configuration for SLNG Text-to-Speech

    SLNG provides unified access to 13 TTS models from providers including
    Deepgram, Cartesia, Fish Audio, Soniox, and others across regional infrastructure.

    Reference: https://docs.slng.ai/api-reference/tts
    """

    # Format mappings from OpenAI to SLNG
    FORMAT_MAPPINGS = {
        "mp3": "mp3",
        "opus": "opus",
        "aac": "aac",
        "flac": "flac",
        "wav": "wav",
        "pcm": "linear16",
    }

    # Sample rate mappings
    SAMPLE_RATE_MAPPINGS = {
        8000: 8000,
        16000: 16000,
        24000: 24000,
        32000: 32000,
        48000: 48000,
    }

    def get_supported_openai_params(self, model: str) -> list:
        """
        SLNG TTS supports these OpenAI parameters
        """
        return ["voice", "response_format", "speed"]

    def map_openai_params(
        self,
        model: str,
        optional_params: dict,
        voice: str | dict | None = None,
        drop_params: bool = False,
        kwargs: dict | None = None,
    ) -> tuple[str | None, dict]:
        """
        Map OpenAI parameters to SLNG TTS parameters

        Args:
            model: Model identifier (e.g., "deepgram/aura:2-en")
            optional_params: OpenAI-style parameters
            voice: Voice identifier (required for SLNG)
            drop_params: Whether to drop unsupported params
            kwargs: Additional passthrough kwargs

        Returns:
            Tuple of (mapped_voice, mapped_params)
        """
        mapped_params: dict[str, Any] = {}
        params = dict(optional_params) if optional_params else {}

        # Extract and validate voice identifier
        mapped_voice: str | None = None
        if isinstance(voice, str) and voice.strip():
            mapped_voice = voice.strip()
        elif isinstance(voice, dict):
            # Support dict with voice_id, id, or name keys
            for key in ("voice_id", "id", "name", "model"):
                candidate = voice.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    mapped_voice = candidate.strip()
                    break

        # Fallback to voice param in optional_params
        if mapped_voice is None:
            voice_override = params.pop("voice", None) or params.pop("voice_id", None)
            if isinstance(voice_override, str) and voice_override.strip():
                mapped_voice = voice_override.strip()

        if mapped_voice is None:
            raise SlngException(
                message="SLNG TTS requires a 'voice' parameter. Pass voice when calling litellm.speech().",
                status_code=400
            )

        # Map response_format (encoding)
        response_format = params.pop("response_format", None)
        if isinstance(response_format, str):
            mapped_format = self.FORMAT_MAPPINGS.get(response_format, response_format)
            mapped_params["encoding"] = mapped_format

        # Map sample_rate if provided
        sample_rate = params.pop("sample_rate", None)
        if sample_rate is not None:
            try:
                rate = int(sample_rate)
                if rate in self.SAMPLE_RATE_MAPPINGS:
                    mapped_params["sample_rate"] = rate
            except (TypeError, ValueError):
                pass

        # Map speed parameter (if SLNG API supports it)
        speed = params.pop("speed", None)
        if speed is not None:
            try:
                speed_value = float(speed)
                # OpenAI supports 0.25 to 4.0 range
                if 0.25 <= speed_value <= 4.0:
                    mapped_params["speed"] = speed_value
            except (TypeError, ValueError):
                pass

        # Handle bit_rate if provided
        bit_rate = params.pop("bit_rate", None)
        if bit_rate is not None:
            try:
                mapped_params["bit_rate"] = int(bit_rate)
            except (TypeError, ValueError):
                pass

        # Handle container format
        container = params.pop("container", None)
        if container is not None:
            mapped_params["container"] = container

        # Drop OpenAI-specific params that SLNG doesn't support
        params.pop("instructions", None)

        # Pass through any extra_body params
        extra_body = params.pop("extra_body", None)
        if isinstance(extra_body, dict):
            for key, value in extra_body.items():
                if value is not None:
                    mapped_params[key] = value

        # Pass through remaining params
        for key, value in params.items():
            if value is not None:
                mapped_params[key] = value

        return mapped_voice, mapped_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        """
        Validate SLNG environment and set up authentication headers

        Args:
            headers: Existing headers dict
            model: Model identifier
            api_key: API key (from params or env)
            api_base: API base URL (from params or env)

        Returns:
            Updated headers dict with authentication
        """
        # Get API key from multiple sources
        resolved_api_key = (
            api_key
            or litellm.api_key
            or get_secret_str("SLNG_API_KEY")
        )

        if not resolved_api_key:
            raise SlngException(
                message="SLNG API key is required. Set SLNG_API_KEY environment variable or pass api_key parameter.",
                status_code=401
            )

        # Update headers with authentication
        headers.update({
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json",
        })

        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        """
        Construct the SLNG TTS endpoint URL

        Args:
            model: Model identifier (e.g., "deepgram/aura:2-en" or full model ID like "aura-2-thalia-en")
            api_base: Base API URL
            litellm_params: LiteLLM parameters (may contain voice info)

        Returns:
            Complete API endpoint URL
        """
        resolved_api_base = get_slng_api_base(api_base)

        # SLNG TTS endpoint structure: POST /v1/tts/slng/{provider}/{model_variant}
        # Example: /v1/tts/slng/deepgram/aura:2-en

        # The model should already be in the format "provider/model:variant"
        # If not, we'll use it as-is and let the API handle it
        endpoint_path = f"/v1/tts/slng/{model}"

        return urljoin(resolved_api_base, endpoint_path)

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> TextToSpeechRequestData:
        """
        Build the SLNG TTS request payload

        Args:
            model: Model identifier
            input: Text to synthesize
            voice: Voice ID (required)
            optional_params: Mapped parameters from map_openai_params
            litellm_params: LiteLLM internal params
            headers: Request headers

        Returns:
            TextToSpeechRequestData with body and headers
        """
        params = dict(optional_params) if optional_params else {}

        # Build request body according to SLNG API spec
        request_body: dict[str, Any] = {
            "text": input,
            "model": voice,  # Voice model ID (e.g., "aura-2-thalia-en")
        }

        # Add optional parameters
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
        Wrap SLNG binary audio response

        Args:
            model: Model identifier
            raw_response: Raw HTTP response from SLNG API
            logging_obj: LiteLLM logging object

        Returns:
            HttpxBinaryResponseContent wrapping the audio bytes
        """
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        return HttpxBinaryResponseContent(raw_response)

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict
    ) -> SlngException:
        """
        Return SLNG-specific exception class

        Args:
            error_message: Error message
            status_code: HTTP status code
            headers: Response headers

        Returns:
            SlngException instance
        """
        return SlngException(
            message=error_message,
            status_code=status_code,
            headers=headers
        )
