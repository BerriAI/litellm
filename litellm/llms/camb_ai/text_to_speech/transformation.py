"""
CAMB AI Text-to-Speech transformation

Maps OpenAI TTS spec to CAMB AI TTS streaming API
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import httpx
from httpx import Headers

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)
from litellm.secret_managers.main import get_secret_str

from ..common_utils import CambAIException

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import HttpxBinaryResponseContent
else:
    LiteLLMLoggingObj = Any
    HttpxBinaryResponseContent = Any


class CambAITextToSpeechConfig(BaseTextToSpeechConfig):
    """
    Configuration for CAMB AI Text-to-Speech

    Reference: https://docs.camb.ai
    """

    TTS_BASE_URL = "https://client.camb.ai/apis"
    TTS_ENDPOINT_PATH = "/tts-stream"

    # Response format mappings from OpenAI to CAMB AI output_configuration
    FORMAT_MAPPINGS = {
        "mp3": "mp3",
        "wav": "wav",
        "pcm": "pcm",
        "flac": "flac",
    }

    def get_supported_openai_params(self, model: str) -> list:
        return ["voice", "response_format", "language"]

    def map_openai_params(
        self,
        model: str,
        optional_params: Dict,
        voice: Optional[Union[str, Dict]] = None,
        drop_params: bool = False,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Dict]:
        mapped_params: Dict[str, Any] = {}
        params = dict(optional_params) if optional_params else {}

        # Extract voice_id — CAMB AI uses integer voice IDs
        voice_id: Optional[str] = None
        if isinstance(voice, str) and voice.strip():
            voice_id = voice.strip()
        elif isinstance(voice, dict):
            for key in ("voice_id", "id", "name"):
                candidate = voice.get(key)
                if isinstance(candidate, (str, int)) and str(candidate).strip():
                    voice_id = str(candidate).strip()
                    break

        if voice_id is not None:
            try:
                mapped_params["voice_id"] = int(voice_id)
            except (TypeError, ValueError):
                mapped_params["voice_id"] = voice_id

        # Response format
        response_format = params.pop("response_format", None)
        if isinstance(response_format, str):
            mapped_format = self.FORMAT_MAPPINGS.get(response_format, response_format)
            mapped_params["output_configuration"] = {"format": mapped_format}

        # Speed — drop silently (CAMB AI doesn't support it directly)
        params.pop("speed", None)

        # Instructions — OpenAI-specific, omit
        params.pop("instructions", None)

        # Pass through remaining params
        for key, value in params.items():
            if value is None:
                continue
            mapped_params[key] = value

        return voice_id, mapped_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        api_key = (
            api_key
            or litellm.api_key
            or get_secret_str("CAMB_API_KEY")
        )

        if api_key is None:
            raise ValueError(
                "CAMB AI API key is required. Set CAMB_API_KEY environment variable."
            )

        headers.update(
            {
                "x-api-key": api_key,
                "Content-Type": "application/json",
            }
        )

        return headers

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, Headers]
    ) -> BaseLLMException:
        return CambAIException(
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
        params = dict(optional_params) if optional_params else {}
        extra_body = params.pop("extra_body", None)

        # Get language from kwargs/litellm_params, default to English (US)
        language = litellm_params.get("language", params.pop("language", "en-us"))

        request_body: Dict[str, Any] = {
            "text": input,
            "language": language,
            "speech_model": model,
        }

        # Apply response_format → output_configuration mapping
        response_format = params.pop("response_format", None)
        if isinstance(response_format, str):
            mapped_format = self.FORMAT_MAPPINGS.get(response_format, response_format)
            request_body["output_configuration"] = {"format": mapped_format}

        # Drop speed silently (CAMB AI doesn't support it)
        params.pop("speed", None)

        # Add voice_id if present
        voice_id = params.pop("voice_id", None)
        if voice_id is not None:
            request_body["voice_id"] = voice_id
        elif voice is not None:
            # Support dict-style voice extraction
            if isinstance(voice, dict):
                for key in ("voice_id", "id", "name"):
                    candidate = voice.get(key)
                    if isinstance(candidate, (str, int)) and str(candidate).strip():
                        voice = str(candidate).strip()
                        break
            try:
                request_body["voice_id"] = int(voice)
            except (TypeError, ValueError):
                request_body["voice_id"] = voice

        # Add remaining params
        for key, value in params.items():
            if value is None:
                continue
            request_body[key] = value

        if isinstance(extra_body, dict):
            for key, value in extra_body.items():
                if value is None:
                    continue
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
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        return HttpxBinaryResponseContent(raw_response)

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        base_url = (
            api_base
            or get_secret_str("CAMB_API_BASE")
            or self.TTS_BASE_URL
        )
        base_url = base_url.rstrip("/")

        return f"{base_url}{self.TTS_ENDPOINT_PATH}"
