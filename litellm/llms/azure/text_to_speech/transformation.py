"""
Azure AVA (Cognitive Services) Text-to-Speech transformation

Maps OpenAI TTS spec to Azure Cognitive Services TTS API
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

import httpx

from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import HttpxBinaryResponseContent
else:
    LiteLLMLoggingObj = Any
    HttpxBinaryResponseContent = Any


class AzureAVATextToSpeechConfig(BaseTextToSpeechConfig):
    """
    Configuration for Azure AVA (Cognitive Services) Text-to-Speech
    
    Reference: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/rest-text-to-speech
    """

    # Voice name mappings from OpenAI voices to Azure voices
    VOICE_MAPPINGS = {
        "alloy": "en-US-JennyNeural",
        "echo": "en-US-GuyNeural",
        "fable": "en-GB-RyanNeural",
        "onyx": "en-US-DavisNeural",
        "nova": "en-US-AmberNeural",
        "shimmer": "en-US-AriaNeural",
    }

    # Response format mappings from OpenAI to Azure
    FORMAT_MAPPINGS = {
        "mp3": "audio-24khz-48kbitrate-mono-mp3",
        "opus": "ogg-48khz-16bit-mono-opus",
        "aac": "audio-24khz-48kbitrate-mono-mp3",  # Azure doesn't have AAC, use MP3
        "flac": "audio-24khz-48kbitrate-mono-mp3",  # Azure doesn't have FLAC, use MP3
        "wav": "riff-24khz-16bit-mono-pcm",
        "pcm": "raw-24khz-16bit-mono-pcm",
    }

    def get_supported_openai_params(self, model: str) -> list:
        """
        Azure AVA TTS supports these OpenAI parameters
        """
        return ["voice", "response_format", "speed"]

    def map_openai_params(
        self,
        model: str,
        optional_params: Dict,
        drop_params: bool,
    ) -> Dict:
        """
        Map OpenAI parameters to Azure AVA TTS parameters
        """
        mapped_params = {}
        
        # Map voice
        if "voice" in optional_params:
            voice = optional_params["voice"]
            # If it's already an Azure voice, use it directly
            if isinstance(voice, str):
                if voice in self.VOICE_MAPPINGS:
                    mapped_params["voice"] = self.VOICE_MAPPINGS[voice]
                else:
                    # Assume it's already an Azure voice name
                    mapped_params["voice"] = voice
        
        # Map response format
        if "response_format" in optional_params:
            format_name = optional_params["response_format"]
            if format_name in self.FORMAT_MAPPINGS:
                mapped_params["output_format"] = self.FORMAT_MAPPINGS[format_name]
            else:
                # Try to use it directly as Azure format
                mapped_params["output_format"] = format_name
        else:
            # Default to MP3
            mapped_params["output_format"] = "audio-24khz-48kbitrate-mono-mp3"
        
        # Map speed (OpenAI: 0.25-4.0, Azure: prosody rate)
        if "speed" in optional_params:
            speed = optional_params["speed"]
            if speed is not None:
                # Convert speed to percentage for Azure SSML
                # OpenAI default is 1.0, so we convert to percentage
                # speed=1.0 -> 0% (default)
                # speed=2.0 -> +100%
                # speed=0.5 -> -50%
                rate_percentage = int((speed - 1.0) * 100)
                mapped_params["rate"] = f"{rate_percentage:+d}%"  # Format with sign
        
        return mapped_params

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
        validated_headers = headers.copy()
        
        # Azure AVA TTS requires either:
        # 1. Ocp-Apim-Subscription-Key header, or
        # 2. Authorization: Bearer <token> header
        
        # We'll use the token-based auth via our token handler
        # The token will be added later in the handler
        
        if api_key:
            # If subscription key is provided, use it directly
            validated_headers["Ocp-Apim-Subscription-Key"] = api_key
        
        # Content-Type for SSML
        validated_headers["Content-Type"] = "application/ssml+xml"
        
        # User-Agent
        validated_headers["User-Agent"] = "litellm"
        
        return validated_headers

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Get the complete URL for Azure AVA TTS request
        
        Azure TTS endpoint format:
        https://{region}.tts.speech.microsoft.com/cognitiveservices/v1
        """
        if api_base is None:
            raise ValueError(
                "api_base is required for Azure AVA TTS. "
                "Format: https://{region}.api.cognitive.microsoft.com or "
                "https://{region}.tts.speech.microsoft.com"
            )
        
        # Remove trailing slash
        api_base = api_base.rstrip("/")
        
        # If it's the general cognitive services endpoint, convert to TTS endpoint
        if "api.cognitive.microsoft.com" in api_base:
            # Extract region from URL
            # e.g., https://eastus.api.cognitive.microsoft.com -> eastus
            region = api_base.split("//")[1].split(".")[0]
            return f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1"
        elif "tts.speech.microsoft.com" in api_base:
            # Already a TTS endpoint
            if not api_base.endswith("/cognitiveservices/v1"):
                return f"{api_base}/cognitiveservices/v1"
            return api_base
        else:
            # Assume it's a custom endpoint, append the path
            return f"{api_base}/cognitiveservices/v1"

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
        Transform OpenAI TTS request to Azure AVA TTS SSML format
        
        Note: optional_params should already be mapped via map_openai_params in main.py
        
        Returns:
            TextToSpeechRequestData: Contains SSML body and Azure-specific headers
        """
        # Get voice (already mapped in main.py, or use default)
        azure_voice = optional_params.get("voice", "en-US-AriaNeural")
        
        # Get output format (already mapped in main.py)
        output_format = optional_params.get(
            "output_format", "audio-24khz-48kbitrate-mono-mp3"
        )
        headers["X-Microsoft-OutputFormat"] = output_format
        
        # Build SSML
        rate = optional_params.get("rate", "0%")
        
        # Escape XML special characters in input text
        escaped_input = (
            input.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )
        
        ssml_body = f"""<speak version='1.0' xml:lang='en-US'>
    <voice name='{azure_voice}'>
        <prosody rate='{rate}'>
            {escaped_input}
        </prosody>
    </voice>
</speak>"""
        
        return {
            "ssml_body": ssml_body,
            "headers": headers,
        }

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> "HttpxBinaryResponseContent":
        """
        Transform Azure AVA TTS response to standard format
        
        Azure returns the audio data directly in the response body
        """
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        # Azure returns audio data directly in the response body
        # Wrap it in HttpxBinaryResponseContent for consistent return type
        return HttpxBinaryResponseContent(raw_response)

