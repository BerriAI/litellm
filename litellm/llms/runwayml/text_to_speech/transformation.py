"""
RunwayML Text-to-Speech transformation

Maps OpenAI TTS spec to RunwayML Text-to-Speech API
"""
import asyncio
import time
from typing import TYPE_CHECKING, Any, Coroutine, Dict, Optional, Tuple, Union

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.constants import (
    RUNWAYML_DEFAULT_API_VERSION,
    RUNWAYML_POLLING_TIMEOUT,
)
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


class RunwayMLTextToSpeechConfig(BaseTextToSpeechConfig):
    """
    Configuration for RunwayML Text-to-Speech
    
    Reference: https://api.dev.runwayml.com/v1/text_to_speech
    """
    
    DEFAULT_BASE_URL: str = "https://api.dev.runwayml.com"
    TTS_ENDPOINT_PATH: str = "v1/text_to_speech"
    DEFAULT_MODEL: str = "eleven_multilingual_v2"
    DEFAULT_VOICE_TYPE: str = "runway-preset"
    DEFAULT_VOICE_PRESET_ID: str = "Bernard"
    
    # Voice mappings from OpenAI voices to RunwayML preset IDs
    # OpenAI voices mapped to similar-sounding RunwayML voices
    VOICE_MAPPINGS = {
        "alloy": "Maya",      # Neutral, balanced female voice
        "echo": "James",      # Male voice
        "fable": "Bernard",   # Warm, storytelling voice
        "onyx": "Vincent",    # Deep male voice
        "nova": "Serene",     # Warm, expressive female voice
        "shimmer": "Ella",    # Clear, friendly female voice
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
        Dispatch method to handle RunwayML TTS requests
        
        This method encapsulates RunwayML-specific credential resolution and parameter handling
        
        Args:
            base_llm_http_handler: The BaseLLMHTTPHandler instance from main.py
        """
        # Resolve api_base from multiple sources
        api_base = (
            api_base
            or litellm_params_dict.get("api_base")
            or litellm.api_base
            or get_secret_str("RUNWAYML_API_BASE")
            or self.DEFAULT_BASE_URL
        )
        
        # Resolve api_key from multiple sources
        api_key = (
            api_key
            or litellm_params_dict.get("api_key")
            or litellm.api_key
            or get_secret_str("RUNWAYML_API_SECRET")
            or get_secret_str("RUNWAYML_API_KEY")
        )
        
        # Convert voice to appropriate format
        voice_param: Optional[Union[str, Dict]] = voice
        if isinstance(voice, str):
            # Keep as string, will be processed in map_openai_params
            voice_param = voice
        elif isinstance(voice, dict):
            # Already in dict format, pass through
            voice_param = voice
        
        litellm_params_dict.update({
            "api_key": api_key,
            "api_base": api_base,
        })
        
        # Call the text_to_speech_handler
        response = base_llm_http_handler.text_to_speech_handler(
            model=model,
            input=input,
            voice=voice_param,
            text_to_speech_provider_config=self,
            text_to_speech_optional_params=optional_params,
            custom_llm_provider="runwayml",
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
        RunwayML TTS supports these OpenAI parameters
        """
        return ["voice"]

    def map_openai_params(
        self,
        model: str,
        optional_params: Dict,
        voice: Optional[Union[str, Dict]] = None,
        drop_params: bool = False,
        kwargs: Dict = {},
    ) -> Tuple[Optional[str], Dict]:
        """
        Map OpenAI parameters to RunwayML TTS parameters
        
        Returns:
            Tuple of (mapped_voice_string, mapped_params)
            
        Note: Since RunwayML requires voice as a dict, we store it in
        mapped_params["runwayml_voice"] and return None for the voice string.
        """
        mapped_params = {}
        
        # Map voice parameter to RunwayML format dict
        voice_dict: Optional[Dict] = None
        if isinstance(voice, str):
            # Check if it's an OpenAI voice name that needs mapping
            if voice in self.VOICE_MAPPINGS:
                preset_id = self.VOICE_MAPPINGS[voice]
                voice_dict = {
                    "type": self.DEFAULT_VOICE_TYPE,
                    "presetId": preset_id,
                }
            else:
                # Assume it's a RunwayML preset ID
                voice_dict = {
                    "type": self.DEFAULT_VOICE_TYPE,
                    "presetId": voice,
                }
        elif isinstance(voice, dict):
            # Already in RunwayML format, use as-is
            voice_dict = voice
        
        # Store the voice dict in optional_params for later use
        if voice_dict is not None:
            mapped_params["runwayml_voice"] = voice_dict
        
        # No other OpenAI params are currently supported by RunwayML TTS
        # (response_format, speed, etc. are not supported)
        
        # Return None for voice string since RunwayML uses dict format
        return None, mapped_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Validate RunwayML environment and set up authentication headers
        """
        validated_headers = headers.copy()
        
        final_api_key = (
            api_key 
            or get_secret_str("RUNWAYML_API_SECRET") 
            or get_secret_str("RUNWAYML_API_KEY")
        )
        
        if not final_api_key:
            raise ValueError("RUNWAYML_API_SECRET or RUNWAYML_API_KEY is not set")
        
        validated_headers["Authorization"] = f"Bearer {final_api_key}"
        validated_headers["X-Runway-Version"] = RUNWAYML_DEFAULT_API_VERSION
        validated_headers["Content-Type"] = "application/json"
        
        return validated_headers

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Get the complete URL for RunwayML TTS request
        """
        complete_url = (
            api_base 
            or get_secret_str("RUNWAYML_API_BASE") 
            or self.DEFAULT_BASE_URL
        )
        
        complete_url = complete_url.rstrip("/")
        return f"{complete_url}/{self.TTS_ENDPOINT_PATH}"

    @staticmethod
    def _check_timeout(start_time: float, timeout_secs: float) -> None:
        """
        Check if operation has timed out.
        
        Args:
            start_time: Start time of the operation
            timeout_secs: Timeout duration in seconds
            
        Raises:
            TimeoutError: If operation has exceeded timeout
        """
        if time.time() - start_time > timeout_secs:
            raise TimeoutError(
                f"RunwayML TTS task polling timed out after {timeout_secs} seconds"
            )

    @staticmethod
    def _check_task_status(response_data: Dict[str, Any]) -> str:
        """
        Check RunwayML task status from response.
        
        RunwayML statuses: PENDING, RUNNING, SUCCEEDED, FAILED, CANCELLED, THROTTLED
        
        Args:
            response_data: JSON response from RunwayML task endpoint
            
        Returns:
            Normalized status string: "running", "succeeded", or raises on failure
            
        Raises:
            ValueError: If task failed or status is unknown
        """
        status = response_data.get("status", "").upper()
        
        verbose_logger.debug(f"RunwayML TTS task status: {status}")
        
        if status == "SUCCEEDED":
            return "succeeded"
        elif status == "FAILED":
            failure_reason = response_data.get("failure", "Unknown error")
            failure_code = response_data.get("failureCode", "unknown")
            raise ValueError(
                f"RunwayML TTS failed: {failure_reason} (code: {failure_code})"
            )
        elif status == "CANCELLED":
            raise ValueError("RunwayML TTS was cancelled")
        elif status in ["PENDING", "RUNNING", "THROTTLED"]:
            return "running"
        else:
            raise ValueError(f"Unknown RunwayML task status: {status}")

    def _poll_task_sync(
        self,
        task_id: str,
        api_base: str,
        headers: Dict[str, str],
        timeout_secs: float = 600,
    ) -> httpx.Response:
        """
        Poll RunwayML task until completion (sync).
        
        RunwayML POST returns immediately with a task that has status PENDING/RUNNING.
        We need to poll GET /v1/tasks/{task_id} until status is SUCCEEDED or FAILED.
        
        Args:
            task_id: The task ID to poll
            api_base: Base URL for RunwayML API
            headers: Request headers (including auth)
            timeout_secs: Total timeout in seconds (default: 600s = 10 minutes)
            
        Returns:
            Final response with completed task
        """
        from litellm.llms.custom_httpx.http_handler import _get_httpx_client

        client = _get_httpx_client()
        start_time = time.time()
        
        # Build task status URL
        api_base = api_base.rstrip("/")
        task_url = f"{api_base}/v1/tasks/{task_id}"
        
        verbose_logger.debug(f"Polling RunwayML TTS task: {task_url}")
        
        while True:
            self._check_timeout(start_time=start_time, timeout_secs=timeout_secs)
            
            # Poll the task status
            response = client.get(url=task_url, headers=headers)
            response.raise_for_status()
            
            response_data = response.json()
            
            # Check task status
            status = self._check_task_status(response_data=response_data)
            
            if status == "succeeded":
                return response
            elif status == "running":
                # Wait before polling again (RunwayML recommends 1-2 second intervals)
                time.sleep(2)

    async def _poll_task_async(
        self,
        task_id: str,
        api_base: str,
        headers: Dict[str, str],
        timeout_secs: float = 600,
    ) -> httpx.Response:
        """
        Poll RunwayML task until completion (async).
        
        Args:
            task_id: The task ID to poll
            api_base: Base URL for RunwayML API
            headers: Request headers (including auth)
            timeout_secs: Total timeout in seconds (default: 600s = 10 minutes)
            
        Returns:
            Final response with completed task
        """
        from litellm.llms.custom_httpx.http_handler import get_async_httpx_client

        client = get_async_httpx_client(llm_provider=litellm.LlmProviders.RUNWAYML)
        start_time = time.time()
        
        # Build task status URL
        api_base = api_base.rstrip("/")
        task_url = f"{api_base}/v1/tasks/{task_id}"
        
        verbose_logger.debug(f"Polling RunwayML TTS task (async): {task_url}")
        
        while True:
            self._check_timeout(start_time=start_time, timeout_secs=timeout_secs)
            
            # Poll the task status
            response = await client.get(url=task_url, headers=headers)
            response.raise_for_status()
            
            response_data = response.json()
            
            # Check task status
            status = self._check_task_status(response_data=response_data)
            
            if status == "succeeded":
                return response
            elif status == "running":
                # Wait before polling again (RunwayML recommends 1-2 second intervals)
                await asyncio.sleep(2)

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: Optional[Union[str, Dict]],
        optional_params: Dict,
        litellm_params: Dict,
        headers: dict,
    ) -> TextToSpeechRequestData:
        """
        Transform OpenAI TTS request to RunwayML TTS format
        
        RunwayML expects:
        - model: The model to use (e.g., 'eleven_multilingual_v2')
        - promptText: The text to convert to speech
        - voice: Voice configuration object
          {
            "type": "runway-preset",
            "presetId": "Bernard"
          }
        
        Returns:
            TextToSpeechRequestData: Contains JSON body and headers
        """
        # Get voice from optional_params (mapped in map_openai_params)
        runwayml_voice = optional_params.get("runwayml_voice")
        if runwayml_voice is None:
            # Use default voice if not provided
            runwayml_voice = {
                "type": self.DEFAULT_VOICE_TYPE,
                "presetId": self.DEFAULT_VOICE_PRESET_ID,
            }
        
        # Build request body
        request_body = {
            "model": model or self.DEFAULT_MODEL,
            "promptText": input,
            "voice": runwayml_voice,
        }
        
        # Add any other optional parameters (except runwayml_voice which we already used)
        for k, v in optional_params.items():
            if k not in request_body and k != "runwayml_voice":
                request_body[k] = v
        
        return {
            "dict_body": request_body,
            "headers": headers,
        }

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> "HttpxBinaryResponseContent":
        """
        Transform RunwayML TTS response to standard format
        
        RunwayML returns a task immediately with status PENDING/RUNNING.
        We need to poll the task until it completes, then download the audio.
        
        Initial response:
        {
            "id": "task_123...",
            "status": "PENDING" | "RUNNING",
            "createdAt": "2025-11-13T..."
        }
        
        After polling:
        {
            "id": "task_123...",
            "status": "SUCCEEDED",
            "output": ["https://storage.googleapis.com/.../audio.mp3"],
            "completedAt": "2025-11-13T..."
        }
        """
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        try:
            response_data = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Error parsing RunwayML TTS response: {e}",
                status_code=raw_response.status_code,
                headers=dict(raw_response.headers),
            )
        
        verbose_logger.debug("RunwayML TTS starting polling...")
        
        # Get task ID
        task_id = response_data.get("id")
        if not task_id:
            raise ValueError("RunwayML TTS response missing task ID")
        
        # Get headers for polling (need auth)
        poll_headers = {
            "Authorization": raw_response.request.headers.get("Authorization", ""),
            "X-Runway-Version": raw_response.request.headers.get(
                "X-Runway-Version", RUNWAYML_DEFAULT_API_VERSION
            ),
        }
        
        # Poll until task completes
        polled_response = self._poll_task_sync(
            task_id=task_id,
            api_base=self.DEFAULT_BASE_URL,
            headers=poll_headers,
            timeout_secs=RUNWAYML_POLLING_TIMEOUT,
        )
        
        # Get the completed task data
        task_data = polled_response.json()
        
        verbose_logger.debug("RunwayML TTS polling complete, downloading audio")
        
        # Get audio URL from output
        output = task_data.get("output", [])
        if not output or not isinstance(output, list) or len(output) == 0:
            raise ValueError("RunwayML TTS response missing audio URL in output")
        
        audio_url = output[0]
        if not isinstance(audio_url, str):
            raise ValueError(f"RunwayML TTS audio URL is not a string: {audio_url}")
        
        # Download the audio file
        from litellm.llms.custom_httpx.http_handler import _get_httpx_client

        client = _get_httpx_client()
        audio_response = client.get(url=audio_url)
        audio_response.raise_for_status()
        
        verbose_logger.debug("RunwayML TTS audio downloaded successfully")
        
        # Return the audio data wrapped in HttpxBinaryResponseContent
        return HttpxBinaryResponseContent(audio_response)

    async def async_transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> "HttpxBinaryResponseContent":
        """
        Async transform RunwayML TTS response to standard format
        
        Same as sync version but uses async polling and download
        """
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        try:
            response_data = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Error parsing RunwayML TTS response: {e}",
                status_code=raw_response.status_code,
                headers=dict(raw_response.headers),
            )
        
        verbose_logger.debug("RunwayML TTS starting polling (async)...")
        
        # Get task ID
        task_id = response_data.get("id")
        if not task_id:
            raise ValueError("RunwayML TTS response missing task ID")
        
        # Get headers for polling (need auth)
        poll_headers = {
            "Authorization": raw_response.request.headers.get("Authorization", ""),
            "X-Runway-Version": raw_response.request.headers.get(
                "X-Runway-Version", RUNWAYML_DEFAULT_API_VERSION
            ),
        }
        
        # Poll until task completes (async)
        polled_response = await self._poll_task_async(
            task_id=task_id,
            api_base=self.DEFAULT_BASE_URL,
            headers=poll_headers,
            timeout_secs=RUNWAYML_POLLING_TIMEOUT,
        )
        
        # Get the completed task data
        task_data = polled_response.json()
        
        verbose_logger.debug("RunwayML TTS polling complete (async), downloading audio")
        
        # Get audio URL from output
        output = task_data.get("output", [])
        if not output or not isinstance(output, list) or len(output) == 0:
            raise ValueError("RunwayML TTS response missing audio URL in output")
        
        audio_url = output[0]
        if not isinstance(audio_url, str):
            raise ValueError(f"RunwayML TTS audio URL is not a string: {audio_url}")
        
        # Download the audio file (async)
        from litellm.llms.custom_httpx.http_handler import get_async_httpx_client

        client = get_async_httpx_client(llm_provider=litellm.LlmProviders.RUNWAYML)
        audio_response = await client.get(url=audio_url)
        audio_response.raise_for_status()
        
        verbose_logger.debug("RunwayML TTS audio downloaded successfully (async)")
        
        # Return the audio data wrapped in HttpxBinaryResponseContent
        return HttpxBinaryResponseContent(audio_response)

