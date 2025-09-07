"""
ElevenLabs Text-to-Speech API handler.
Following Vertex AI text_to_speech pattern.
"""

from typing import Optional, Union
import httpx

import litellm
from litellm.llms.openai.openai import HttpxBinaryResponseContent
from litellm.llms.custom_httpx.http_handler import _get_httpx_client
from litellm import get_secret_str

class ElevenLabsTextToSpeechAPI:
    """
    ElevenLabs TTS API methods.
    """

    def __init__(self) -> None:
        pass

    def audio_speech(
        self,
        logging_obj,
        model: str,
        input: str,
        voice: str,
        optional_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: Union[float, httpx.Timeout] = 60,
        aspeech: Optional[bool] = None,
        litellm_params: Optional[dict] = None,
    ) -> HttpxBinaryResponseContent:
        """
        Convert text to speech using ElevenLabs API.

        Args:
            logging_obj: Logger object
            model: ElevenLabs model (e.g., "eleven_monolingual_v1")
            input: Text to convert to speech
            voice: ElevenLabs voice ID
            optional_params: Additional parameters (response_format, speed)
            api_key: ElevenLabs API key
            api_base: API base URL
            timeout: Request timeout
            aspeech: Whether to use async (not supported yet)
            litellm_params: LiteLLM specific parameters

        Returns:
            HttpxBinaryResponseContent: Audio response
        """
        ####### Authenticate with ElevenLabs ########
        if not api_key or not isinstance(api_key, str):
            raise litellm.AuthenticationError(
                message="ElevenLabs API key is required. Set ELEVENLABS_API_KEY environment variable.",
                llm_provider="elevenlabs",
                model=model,
            )

        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        }
        ######### End of Authentication ###########

        ####### Build the request ################
        # API Ref: https://elevenlabs.io/docs/api-reference/text-to-speech/convert

        # Handle output format as query parameter
        output_format = optional_params.get("output_format")
        if output_format is None:
            if response_format := optional_params.get("response_format"):
                # If user provided OpenAI style response_format, map to ElevenLabs format
                format_map = {
                    "mp3": "mp3_44100_128",
                    "pcm": "pcm_44100",
                    "opus": "opus_48000_128",
                    # wav, aac, flac not supported by ElevenLabs
                }
                if response_format in format_map:
                    output_format = format_map[response_format]

        url = self._get_complete_url(
            voice=voice,
            api_base=api_base,
            output_format=output_format,
        )

        # Build request body
        request_data: dict = {
            "text": input,
            "model_id": model,
        }

        # Map speed to voice settings
        if optional_params.get("speed"):
            speed = float(optional_params["speed"])
            # ElevenLabs supports direct speed parameter in voice_settings
            voice_settings = {
                "speed": speed
            }
            request_data["voice_settings"] = voice_settings
        ########## End of building request ############

        ########## Log the request for debugging / logging ############
        logging_obj.pre_call(
            input=[],
            api_key="",
            additional_args={
                "complete_input_dict": request_data,
                "api_base": url,
                "headers": headers,
            },
        )
        ########## End of logging ############

        ####### Send the request ###################
        if aspeech:
            pass

        # Send HTTP request
        sync_handler = _get_httpx_client()

        response = sync_handler.post(
            url=url,
            headers=headers,
            json=request_data,
            timeout=timeout,
        )

        ############ Process the error message ############
        if response.status_code != 200:
            error_msg = f"ElevenLabs API error: {response.status_code}"
            try:
                error_data = response.json()
                if "detail" in error_data:
                    error_msg += f" - {error_data['detail']}"
            except:
                error_msg += f" - {response.text}"

            raise litellm.APIError(
                status_code=response.status_code,
                message=error_msg,
                llm_provider="elevenlabs",
                model=model,
            )

        ############ Process the response ############
        # Create HttpxBinaryResponseContent from the binary audio data
        audio_response = httpx.Response(
            status_code=200,
            content=response.content,
        )

        return HttpxBinaryResponseContent(response=audio_response)

    def _get_complete_url(
        self,
        voice: str,
        api_base: Optional[str],
        output_format: Optional[str] = None,
    ) -> str:
        if api_base is None:
            api_base = (
                get_secret_str("ELEVENLABS_API_BASE") or "https://api.elevenlabs.io"
            )
        api_base = api_base.rstrip("/")  # Remove trailing slash if present

        # ElevenLabs text-to-speech endpoint
        url = f"{api_base}/v1/text-to-speech/{voice}"

        if output_format is None:
            return url

        return f"{url}?output_format={output_format}"
