"""
Elevenlabs Text-to-Speech transformation

Maps OpenAI TTS spec to Elevenlabs TTS API
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
from urllib.parse import urlencode

import httpx
from httpx import Headers

import litellm
from litellm.types.utils import all_litellm_params
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)
from litellm.secret_managers.main import get_secret_str

from ..common_utils import ElevenLabsException


if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import HttpxBinaryResponseContent
else:
    LiteLLMLoggingObj = Any
    HttpxBinaryResponseContent = Any


class ElevenLabsTextToSpeechConfig(BaseTextToSpeechConfig):
    """
    Configuration for ElevenLabs Text-to-Speech

    Reference: https://elevenlabs.io/docs/api-reference/text-to-speech/convert
    """

    TTS_BASE_URL = "https://api.elevenlabs.io"
    TTS_ENDPOINT_PATH = "/v1/text-to-speech"
    DEFAULT_OUTPUT_FORMAT = "pcm_44100"
    VOICE_MAPPINGS = {
        "alloy": "21m00Tcm4TlvDq8ikWAM",  # Rachel
        "amber": "5Q0t7uMcjvnagumLfvZi",  # Paul
        "ash": "AZnzlk1XvdvUeBnXmlld",  # Domi
        "august": "D38z5RcWu1voky8WS1ja",  # Fin
        "blue": "2EiwWnXFnvU5JabPnv8n",  # Clyde
        "coral": "9BWtsMINqrJLrRacOk9x",  # Aria
        "lily": "EXAVITQu4vr4xnSDxMaL",  # Sarah
        "onyx": "29vD33N1CtxCmqQRPOHJ",  # Drew
        "sage": "CwhRBWXzGAHq8TQ4Fs17",  # Roger
        "verse": "CYw3kZ02Hs0563khs1Fj",  # Dave
    }

    # Response format mappings from OpenAI to ElevenLabs
    FORMAT_MAPPINGS = {
        "mp3": "mp3_44100_128",
        "pcm": "pcm_44100",
        "opus": "opus_48000_128",
        # ElevenLabs does not support WAV, AAC, or FLAC formats.
    }

    ELEVENLABS_QUERY_PARAMS_KEY = "__elevenlabs_query_params__"
    ELEVENLABS_VOICE_ID_KEY = "__elevenlabs_voice_id__"

    def get_supported_openai_params(self, model: str) -> list:
        """
        ElevenLabs TTS supports these OpenAI parameters
        """
        return ["voice", "response_format", "speed"]

    def _extract_voice_id(self, voice: str) -> str:
        """
        Normalize the provided voice information into an ElevenLabs voice_id.
        """
        normalized_voice = voice.strip()
        mapped_voice = self.VOICE_MAPPINGS.get(normalized_voice.lower())
        return mapped_voice or normalized_voice

    def _resolve_voice_id(
        self,
        voice: Optional[Union[str, Dict[str, Any]]],
        params: Dict[str, Any],
    ) -> str:
        """
        Determine the ElevenLabs voice_id based on provided voice input or parameters.
        """
        mapped_voice: Optional[str] = None

        if isinstance(voice, str) and voice.strip():
            mapped_voice = self._extract_voice_id(voice)
        elif isinstance(voice, dict):
            for key in ("voice_id", "id", "name"):
                candidate = voice.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    mapped_voice = self._extract_voice_id(candidate)
                    break
        elif voice is not None:
            mapped_voice = self._extract_voice_id(str(voice))

        if mapped_voice is None:
            voice_override = params.pop("voice_id", None)
            if isinstance(voice_override, str) and voice_override.strip():
                mapped_voice = self._extract_voice_id(voice_override)

        if mapped_voice is None:
            raise ValueError(
                "ElevenLabs voice_id is required. Pass `voice` when calling `litellm.speech()`."
            )

        return mapped_voice

    def map_openai_params(
        self,
        model: str,
        optional_params: Dict,
        voice: Optional[Union[str, Dict]] = None,
        drop_params: bool = False,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[str], Dict]:
        """
        Map OpenAI parameters to ElevenLabs TTS parameters
        """
        mapped_params: Dict[str, Any] = {}
        query_params: Dict[str, Any] = {}

        # Work on a copy so we don't mutate the caller's dictionary
        params = dict(optional_params) if optional_params else {}
        passthrough_kwargs: Dict[str, Any] = kwargs if kwargs is not None else {}

        # Extract voice identifier
        mapped_voice = self._resolve_voice_id(voice, params)

        # Response/output format → query parameter
        response_format = params.pop("response_format", None)
        if isinstance(response_format, str):
            mapped_format = self.FORMAT_MAPPINGS.get(response_format, response_format)
            query_params["output_format"] = mapped_format

        # ElevenLabs does not support OpenAI speed directly.
        # Drop it to avoid sending unsupported keys unless caller already provided voice_settings.
        speed = params.pop("speed", None)
        if speed is not None:
            speed_value: Optional[float]
            try:
                speed_value = float(speed)
            except (TypeError, ValueError):
                speed_value = None
            if speed_value is not None:
                if isinstance(params.get("voice_settings"), dict):
                    params["voice_settings"]["speed"] = speed_value  # type: ignore[index]
                else:
                    params["voice_settings"] = {"speed": speed_value}

        # Instructions parameter is OpenAI-specific; omit to prevent API errors.
        params.pop("instructions", None)
        self._add_elevenlabs_specific_params(
            mapped_voice=mapped_voice,
            query_params=query_params,
            mapped_params=mapped_params,
            kwargs=passthrough_kwargs,
            remaining_params=params,
        )

        return mapped_voice, mapped_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Validate Azure environment and set up authentication headers
        """
        api_key = (
            api_key
            or litellm.api_key
            or litellm.openai_key
            or get_secret_str("ELEVENLABS_API_KEY")
        )

        if api_key is None:
            raise ValueError(
                "ElevenLabs API key is required. Set ELEVENLABS_API_KEY environment variable."
            )

        headers.update(
            {
                "xi-api-key": api_key,
                "Content-Type": "application/json",
            }
        )        
        
        return headers
    
    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, Headers]
    ) -> BaseLLMException:
        return ElevenLabsException(
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
        Build the ElevenLabs TTS request payload.
        """
        params = dict(optional_params) if optional_params else {}
        extra_body = params.pop("extra_body", None)

        request_body: Dict[str, Any] = {
            "text": input,
            "model_id": model,
        }

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

    def _add_elevenlabs_specific_params(
        self,
        mapped_voice: str,
        query_params: Dict[str, Any],
        mapped_params: Dict[str, Any],
        kwargs: Optional[Dict[str, Any]],
        remaining_params: Dict[str, Any],
    ) -> None:
        if kwargs is None:
            kwargs = {}
        for key, value in remaining_params.items():
            if value is None:
                continue
            mapped_params[key] = value

        reserved_kwarg_keys = set(all_litellm_params) | {
            self.ELEVENLABS_QUERY_PARAMS_KEY,
            self.ELEVENLABS_VOICE_ID_KEY,
            "voice",
            "model",
            "response_format",
            "output_format",
            "extra_body",
            "user",
        }

        extra_body_from_kwargs = kwargs.pop("extra_body", None)
        if isinstance(extra_body_from_kwargs, dict):
            for key, value in extra_body_from_kwargs.items():
                if value is None:
                    continue
                mapped_params[key] = value

        for key in list(kwargs.keys()):
            if key in reserved_kwarg_keys:
                continue
            value = kwargs[key]
            if value is None:
                continue
            mapped_params[key] = value
            kwargs.pop(key, None)

        if query_params:
            kwargs[self.ELEVENLABS_QUERY_PARAMS_KEY] = query_params
        else:
            kwargs.pop(self.ELEVENLABS_QUERY_PARAMS_KEY, None)

        kwargs[self.ELEVENLABS_VOICE_ID_KEY] = mapped_voice

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> "HttpxBinaryResponseContent":
        """
        Wrap ElevenLabs binary audio response.
        """
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        return HttpxBinaryResponseContent(raw_response)

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Construct the ElevenLabs endpoint URL, including path voice_id and query params.
        """
        base_url = (
            api_base
            or get_secret_str("ELEVENLABS_API_BASE")
            or self.TTS_BASE_URL
        )
        base_url = base_url.rstrip("/")

        voice_id = litellm_params.get(self.ELEVENLABS_VOICE_ID_KEY)
        if not isinstance(voice_id, str) or not voice_id.strip():
            raise ValueError(
                "ElevenLabs voice_id is required. Pass `voice` when calling `litellm.speech()`."
            )

        url = f"{base_url}{self.TTS_ENDPOINT_PATH}/{voice_id}"

        query_params = litellm_params.get(self.ELEVENLABS_QUERY_PARAMS_KEY, {})
        if query_params:
            url = f"{url}?{urlencode(query_params)}"

        return url