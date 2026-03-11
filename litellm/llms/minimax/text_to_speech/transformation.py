"""
MiniMax Text-to-Speech transformation

Maps OpenAI TTS spec to MiniMax TTS API (WebSocket-based HTTP API)
Reference: https://platform.minimax.io/docs
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

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import HttpxBinaryResponseContent
else:
    LiteLLMLoggingObj = Any
    HttpxBinaryResponseContent = Any


class MinimaxException(BaseLLMException):
    """Custom exception for MiniMax API errors"""

    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Optional[Union[dict, Headers]] = None,
    ):
        super().__init__(status_code=status_code, message=message, headers=headers)


class MinimaxTextToSpeechConfig(BaseTextToSpeechConfig):
    """
    Configuration for MiniMax Text-to-Speech

    Reference: https://platform.minimax.io/docs
    
    MiniMax TTS API supports both WebSocket and HTTP endpoints.
    This implementation uses the HTTP endpoint for simplicity.
    """

    TTS_BASE_URL = "https://api.minimax.io"
    TTS_ENDPOINT_PATH = "/v1/t2a_v2"

    # Voice mappings from OpenAI-style voices to MiniMax voice IDs
    # MiniMax supports many voices, these are common mappings
    VOICE_MAPPINGS = {
        "alloy": "male-qn-qingse",
        "echo": "male-qn-jingying",
        "fable": "female-shaonv",
        "onyx": "male-qn-badao",
        "nova": "female-yujie",
        "shimmer": "female-tianmei",
    }

    # Response format mappings from OpenAI to MiniMax
    FORMAT_MAPPINGS = {
        "mp3": "mp3",
        "pcm": "pcm",
        "wav": "wav",
        "flac": "flac",
    }

    def get_supported_openai_params(self, model: str) -> list:
        """
        MiniMax TTS supports these OpenAI parameters
        """
        return ["voice", "response_format", "speed"]

    def _extract_voice_id(self, voice: str) -> str:
        """
        Normalize the provided voice information into a MiniMax voice_id.
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
        Determine the MiniMax voice_id based on provided voice input or parameters.
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
            # Default to a common voice if not specified
            mapped_voice = "male-qn-qingse"

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
        Map OpenAI parameters to MiniMax TTS parameters
        """
        mapped_params: Dict[str, Any] = {}

        # Work on a copy so we don't mutate the caller's dictionary
        params = dict(optional_params) if optional_params else {}

        # Extract voice identifier
        mapped_voice = self._resolve_voice_id(voice, params)

        # Response/output format
        response_format = params.pop("response_format", None)
        if isinstance(response_format, str):
            mapped_format = self.FORMAT_MAPPINGS.get(response_format, "mp3")
            mapped_params["format"] = mapped_format
        else:
            mapped_params["format"] = "mp3"  # Default format

        # Speed parameter (MiniMax supports speed from 0.5 to 2.0)
        speed = params.pop("speed", None)
        if speed is not None:
            try:
                speed_value = float(speed)
                # Clamp speed to MiniMax's supported range
                speed_value = max(0.5, min(2.0, speed_value))
                mapped_params["speed"] = speed_value
            except (TypeError, ValueError):
                mapped_params["speed"] = 1.0
        else:
            mapped_params["speed"] = 1.0

        # Instructions parameter is OpenAI-specific; omit to prevent API errors
        params.pop("instructions", None)

        # Store voice_id for later use in request construction
        mapped_params["voice_id"] = mapped_voice

        # Handle extra_body for additional MiniMax-specific parameters
        extra_body = params.pop("extra_body", None)
        if isinstance(extra_body, dict):
            for key, value in extra_body.items():
                if value is not None:
                    mapped_params[key] = value

        # Pass through any remaining parameters
        for key, value in params.items():
            if value is not None:
                mapped_params[key] = value

        return mapped_voice, mapped_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Validate MiniMax environment and set up authentication headers
        """
        api_key = (
            api_key
            or litellm.api_key
            or get_secret_str("MINIMAX_API_KEY")
        )

        if api_key is None:
            raise ValueError(
                "MiniMax API key is required. Set MINIMAX_API_KEY environment variable or pass api_key parameter."
            )

        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

        return headers

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, Headers]
    ) -> BaseLLMException:
        return MinimaxException(
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
        Build the MiniMax TTS request payload.
        
        MiniMax uses a different structure than OpenAI:
        - model: The TTS model to use
        - text: The input text
        - voice_setting: Voice configuration
        - audio_setting: Audio output configuration
        """
        params = dict(optional_params) if optional_params else {}

        # Extract parameters
        voice_id = params.pop("voice_id", voice or "male-qn-qingse")
        speed = params.pop("speed", 1.0)
        audio_format = params.pop("format", "mp3")
        
        # Extract additional voice settings
        vol = params.pop("vol", 1.0)  # Volume (0.1 to 10)
        pitch = params.pop("pitch", 0)  # Pitch adjustment (-12 to 12)
        
        # Extract audio settings
        sample_rate = params.pop("sample_rate", 32000)  # 16000, 24000, 32000
        bitrate = params.pop("bitrate", 128000)  # For MP3: 64000, 128000, 192000, 256000
        channel = params.pop("channel", 1)  # 1 for mono, 2 for stereo
        
        # Output format: 'url' or 'hex' (default is 'hex')
        output_format = params.pop("output_format", "hex")

        request_body: Dict[str, Any] = {
            "model": model,
            "text": input,
            "stream": False,  # HTTP endpoint doesn't support streaming
            "output_format": output_format,  # 'url' or 'hex'
            "voice_setting": {
                "voice_id": voice_id,
                "speed": speed,
                "vol": vol,
                "pitch": pitch,
            },
            "audio_setting": {
                "sample_rate": sample_rate,
                "bitrate": bitrate,
                "format": audio_format,
                "channel": channel,
            },
        }

        # Handle any remaining parameters from extra_body
        extra_body = params.pop("extra_body", None)
        if isinstance(extra_body, dict):
            for key, value in extra_body.items():
                if value is not None and key not in request_body:
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
        Transform MiniMax response to standard format.
        
        MiniMax returns JSON with base64-encoded audio data:
        {
            "base_resp": {"status_code": 0, "status_msg": "success"},
            "audio_file": "<base64_encoded_audio>",
            "extra_info": {...}
        }
        
        We need to decode the base64 audio and return it as binary content.
        """
        import base64
        import json

        from litellm.types.llms.openai import HttpxBinaryResponseContent

        try:
            # Parse JSON response
            response_json = raw_response.json()
            
            # MiniMax API response format check
            # The API can return different structures:
            # 1. {"data": {"audio": "..."}, "status": 0, ...} for HTTP endpoint
            # 2. {"base_resp": {"status_code": 0, ...}, "audio_file": "..."} for older versions
            
            # Check for errors - MiniMax uses "status" field in HTTP endpoint response
            # status: 0 = success, 2 = invalid api key, etc.
            status = response_json.get("status")
            if status is not None and status != 0:
                ced = response_json.get("ced", "Unknown error")
                error_detail = ced if ced else f"API returned status {status}"
                raise MinimaxException(
                    status_code=raw_response.status_code,
                    message=f"MiniMax TTS error: {error_detail}",
                    headers=dict(raw_response.headers),
                )
            
            # Extract audio data
            # MiniMax returns audio in "data" field
            data = response_json.get("data", {})
            
            # Check if response contains a URL (output_format='url')
            audio_url = data.get("audio_url", None)
            if audio_url:
                # If URL format is used, we need to fetch the audio from the URL
                # For now, return a response indicating URL mode (TODO: fetch audio from URL)
                raise MinimaxException(
                    status_code=500,
                    message=f"URL output format is not yet supported. Use 'hex' format or fetch from URL: {audio_url}",
                    headers=dict(raw_response.headers),
                )
            
            # Get hex-encoded audio data
            audio_hex = data.get("audio", "") or response_json.get("audio_file", "")
            
            if not audio_hex:
                raise MinimaxException(
                    status_code=500,
                    message=f"No audio data in MiniMax response. Response keys: {list(response_json.keys())}",
                    headers=dict(raw_response.headers),
                )
            
            # MiniMax returns hex-encoded audio by default
            # Try hex decoding first, fall back to base64 if that fails
            try:
                audio_bytes = bytes.fromhex(audio_hex)
            except ValueError:
                # If hex decoding fails, try base64 (for older API versions)
                try:
                    audio_bytes = base64.b64decode(audio_hex)
                except Exception as e:
                    raise MinimaxException(
                        status_code=500,
                        message=f"Failed to decode audio data: {str(e)}",
                        headers=dict(raw_response.headers),
                    )
            
            # Create a new response with binary audio content
            # We need to create a response that contains the decoded audio bytes
            # Remove gzip encoding headers to avoid decompression issues
            clean_headers = dict(raw_response.headers)
            clean_headers.pop('content-encoding', None)
            clean_headers.pop('transfer-encoding', None)
            clean_headers['content-length'] = str(len(audio_bytes))
            
            # Create a new response object with the binary content
            binary_response = httpx.Response(
                status_code=200,
                headers=clean_headers,
                content=audio_bytes,
                request=raw_response.request,
            )
            
            return HttpxBinaryResponseContent(binary_response)
            
        except json.JSONDecodeError as e:
            raise MinimaxException(
                status_code=500,
                message=f"Failed to parse MiniMax response: {str(e)}",
                headers=dict(raw_response.headers),
            )
        except Exception as e:
            if isinstance(e, MinimaxException):
                raise
            raise MinimaxException(
                status_code=500,
                message=f"Error processing MiniMax response: {str(e)}",
                headers=dict(raw_response.headers),
            )

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Construct the MiniMax endpoint URL.
        """
        base_url = (
            api_base
            or get_secret_str("MINIMAX_API_BASE")
            or self.TTS_BASE_URL
        )
        base_url = base_url.rstrip("/")

        # MiniMax uses a simple endpoint path
        url = f"{base_url}{self.TTS_ENDPOINT_PATH}"

        return url

