"""
Azure AVA (Cognitive Services) Text-to-Speech transformation

Maps OpenAI TTS spec to Azure Cognitive Services TTS API
"""

from typing import TYPE_CHECKING, Any, Coroutine, Dict, Optional, Union
from urllib.parse import urlparse

import httpx

import litellm
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


class AzureAVATextToSpeechConfig(BaseTextToSpeechConfig):
    """
    Configuration for Azure AVA (Cognitive Services) Text-to-Speech
    
    Reference: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/rest-text-to-speech
    """

    # Azure endpoint domains
    COGNITIVE_SERVICES_DOMAIN = "api.cognitive.microsoft.com"
    TTS_SPEECH_DOMAIN = "tts.speech.microsoft.com"
    TTS_ENDPOINT_PATH = "/cognitiveservices/v1"

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

    def dispatch_text_to_speech(
        self,
        model: str,
        input: str,
        voice: Optional[Union[str, Dict]],
        optional_params: Dict,
        litellm_params_dict: Dict,
        logging_obj: "LiteLLMLoggingObj",
        timeout: Union[float, httpx.Timeout],
        extra_headers: Optional[Dict[str, Any]],
        base_llm_http_handler: Any,
        aspeech: bool,
        api_base: Optional[str],
        api_key: Optional[str],
        **kwargs: Any,
    ) -> Union[
        "HttpxBinaryResponseContent",
        Coroutine[Any, Any, "HttpxBinaryResponseContent"],
    ]:
        """
        Dispatch method to handle Azure AVA TTS requests
        
        This method encapsulates Azure-specific credential resolution and parameter handling
        
        Args:
            base_llm_http_handler: The BaseLLMHTTPHandler instance from main.py
        """
        # Resolve api_base from multiple sources
        api_base = (
            api_base
            or litellm_params_dict.get("api_base")
            or litellm.api_base
            or get_secret_str("AZURE_API_BASE")
        )
        
        # Resolve api_key from multiple sources (Azure-specific)
        api_key = (
            api_key
            or litellm_params_dict.get("api_key")
            or litellm.api_key
            or litellm.azure_key
            or get_secret_str("AZURE_OPENAI_API_KEY")
            or get_secret_str("AZURE_API_KEY")
        )
        
        # Convert voice to string if it's a dict (for Azure AVA, voice must be a string)
        voice_str: Optional[str] = None
        if isinstance(voice, str):
            voice_str = voice
        elif isinstance(voice, dict):
            # Extract voice name from dict if needed
            voice_str = voice.get("name") if voice else None
        
        litellm_params_dict.update({
            "api_key": api_key,
            "api_base": api_base,
        })
        # Call the text_to_speech_handler
        response = base_llm_http_handler.text_to_speech_handler(
            model=model,
            input=input,
            voice=voice_str,
            text_to_speech_provider_config=self,
            text_to_speech_optional_params=optional_params,
            custom_llm_provider="azure",
            litellm_params=litellm_params_dict,
            logging_obj=logging_obj,
            timeout=timeout,
            extra_headers=extra_headers,
            client=None,
            _is_async=aspeech,
        )
        
        return response

    def get_supported_openai_params(self, model: str) -> list:
        """
        Azure AVA TTS supports these OpenAI parameters
        """
        return ["voice", "response_format", "speed"]

    def _convert_speed_to_azure_rate(self, speed: float) -> str:
        """
        Convert OpenAI speed value to Azure SSML prosody rate percentage
        
        Args:
            speed: OpenAI speed value (0.25-4.0, default 1.0)
        
        Returns:
            Azure rate string with percentage (e.g., "+50%", "-50%", "+0%")
        
        Examples:
            speed=1.0 -> "+0%" (default)
            speed=2.0 -> "+100%"
            speed=0.5 -> "-50%"
        """
        rate_percentage = int((speed - 1.0) * 100)
        return f"{rate_percentage:+d}%"

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
                mapped_params["rate"] = self._convert_speed_to_azure_rate(speed=speed)
        
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
                f"api_base is required for Azure AVA TTS. "
                f"Format: https://{{region}}.{self.COGNITIVE_SERVICES_DOMAIN} or "
                f"https://{{region}}.{self.TTS_SPEECH_DOMAIN}"
            )
        
        # Remove trailing slash and parse URL
        api_base = api_base.rstrip("/")
        parsed_url = urlparse(api_base)
        hostname = parsed_url.hostname or ""
        
        # Check if it's a Cognitive Services endpoint (convert to TTS endpoint)
        if self._is_cognitive_services_endpoint(hostname=hostname):
            region = self._extract_region_from_hostname(
                hostname=hostname, 
                domain=self.COGNITIVE_SERVICES_DOMAIN
            )
            return self._build_tts_url(region=region)
        
        # Check if it's already a TTS endpoint
        if self._is_tts_endpoint(hostname=hostname):
            if not api_base.endswith(self.TTS_ENDPOINT_PATH):
                return f"{api_base}{self.TTS_ENDPOINT_PATH}"
            return api_base
        
        # Assume it's a custom endpoint, append the path
        return f"{api_base}{self.TTS_ENDPOINT_PATH}"

    def _is_cognitive_services_endpoint(self, hostname: str) -> bool:
        """Check if hostname is a Cognitive Services endpoint"""
        return (
            hostname == self.COGNITIVE_SERVICES_DOMAIN 
            or hostname.endswith(f".{self.COGNITIVE_SERVICES_DOMAIN}")
        )

    def _is_tts_endpoint(self, hostname: str) -> bool:
        """Check if hostname is a TTS endpoint"""
        return (
            hostname == self.TTS_SPEECH_DOMAIN 
            or hostname.endswith(f".{self.TTS_SPEECH_DOMAIN}")
        )

    def _extract_region_from_hostname(self, hostname: str, domain: str) -> str:
        """
        Extract region from hostname
        
        Examples:
            eastus.api.cognitive.microsoft.com -> eastus
            api.cognitive.microsoft.com -> ""
        """
        if hostname.endswith(f".{domain}"):
            return hostname[:-len(f".{domain}")]
        return ""

    def _build_tts_url(self, region: str) -> str:
        """Build the complete TTS URL with region"""
        if region:
            return f"https://{region}.{self.TTS_SPEECH_DOMAIN}{self.TTS_ENDPOINT_PATH}"
        return f"https://{self.TTS_SPEECH_DOMAIN}{self.TTS_ENDPOINT_PATH}"

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
        
        ssml_body = f"""
        <speak version='1.0' xml:lang='en-US'>
            <voice name='{azure_voice}'>
                <prosody rate='{rate}'>
                    {escaped_input}
                </prosody>
            </voice>
        </speak>
        """
        
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

