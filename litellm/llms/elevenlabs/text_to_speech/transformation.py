"""
Transforms OpenAI `/audio/speech` requests to ElevenLabs `/v1/text-to-speech/{voice_id}` format
"""

from typing import List, Optional, Union

import httpx

import litellm
from litellm.llms.base_llm.audio_speech.transformation import BaseAudioSpeechConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import HttpxBinaryResponseContent

from ..common_utils import ElevenLabsException


class ElevenLabsTextToSpeechConfig(BaseAudioSpeechConfig):
    @property
    def custom_llm_provider(self) -> str:
        return litellm.LlmProviders.ELEVENLABS.value

    def get_supported_openai_params(self, model: str) -> List[str]:
        return [
            "voice",
            "response_format",
            "speed",
        ]

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
                if k == "voice":
                    # ElevenLabs requires voice_id in the URL path, not body
                    optional_params["voice_id"] = v
                elif k == "response_format":
                    # Map OpenAI response_format to ElevenLabs output_format
                    # OpenAI: mp3, opus, aac, flac, wav, pcm
                    # ElevenLabs: mp3_44100_128, pcm_16000, etc.
                    format_mapping = {
                        "mp3": "mp3_44100_128",
                        "wav": "pcm_16000", 
                        "pcm": "pcm_16000",
                        "opus": "mp3_44100_128",  # fallback to mp3
                        "aac": "mp3_44100_128",   # fallback to mp3
                        "flac": "mp3_44100_128",  # fallback to mp3
                    }
                    optional_params["output_format"] = format_mapping.get(v, "mp3_44100_128")
                elif k == "speed":
                    # ElevenLabs doesn't have direct speed control in the same way
                    # We'll pass it as is for provider-specific use
                    optional_params["speed"] = v
        
        return optional_params

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> ElevenLabsException:
        return ElevenLabsException(
            message=error_message, status_code=status_code, headers=headers
        )

    def transform_audio_speech_request(
        self,
        model: str,
        input: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform OpenAI-style request to ElevenLabs format
        """
        # Extract voice_id for URL path
        voice_id = optional_params.get("voice_id", "21m00Tcm4TlvDq8ikWAM")  # Default voice
        
        # Build request body for ElevenLabs API
        request_body = {
            "text": input,
        }
        
        # Add optional parameters
        if "output_format" in optional_params:
            # This will be added as query parameter in get_complete_url
            pass
            
        # Add model_id if specified
        if model and model != "elevenlabs/":
            # Remove elevenlabs/ prefix if present
            model_clean = model.replace("elevenlabs/", "")
            if model_clean:
                request_body["model_id"] = model_clean
        
        # Add ElevenLabs-specific voice settings if provided
        if "voice_settings" in optional_params:
            request_body["voice_settings"] = optional_params["voice_settings"]
        
        # Store voice_id in optional_params for URL construction
        optional_params["_voice_id"] = voice_id
        
        return request_body

    def transform_audio_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
    ) -> HttpxBinaryResponseContent:
        """
        ElevenLabs returns audio content directly, so we wrap it appropriately
        """
        if raw_response.status_code != 200:
            raise self.get_error_class(
                error_message=f"ElevenLabs TTS request failed: {raw_response.text}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        
        # Create HttpxBinaryResponseContent from the response
        return HttpxBinaryResponseContent(raw_response)

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
        api_base = api_base.rstrip("/")
        
        # Get voice_id from optional_params
        voice_id = optional_params.get("_voice_id", "21m00Tcm4TlvDq8ikWAM")
        
        # Build URL with voice_id
        url = f"{api_base}/v1/text-to-speech/{voice_id}"
        
        # Add query parameters if needed
        query_params = []
        if "output_format" in optional_params:
            query_params.append(f"output_format={optional_params['output_format']}")
        
        if query_params:
            url += "?" + "&".join(query_params)
        
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