"""
Vertex AI Text-to-Speech transformation

Maps OpenAI TTS spec to Google Cloud Text-to-Speech API
Reference: https://cloud.google.com/text-to-speech/docs/reference/rest/v1/text/synthesize
"""

import base64
from typing import TYPE_CHECKING, Any, Coroutine, Union

import httpx

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)
from litellm.llms.vertex_ai.common_utils import (
    VertexAILyriaModelInfo,
    get_vertex_ai_lyria_model_info,
)
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES
from litellm.types.llms.vertex_ai_text_to_speech import (
    VertexTextToSpeechAudioConfig,
    VertexTextToSpeechInput,
    VertexTextToSpeechVoice,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import HttpxBinaryResponseContent
else:
    LiteLLMLoggingObj = Any
    HttpxBinaryResponseContent = Any


class VertexAITextToSpeechConfig(BaseTextToSpeechConfig, VertexBase):
    """
    Configuration for Google Cloud/Vertex AI Text-to-Speech

    Reference: https://cloud.google.com/text-to-speech/docs/reference/rest/v1/text/synthesize
    """

    # Default values
    DEFAULT_LANGUAGE_CODE = "en-US"
    DEFAULT_VOICE_NAME = "en-US-Studio-O"
    DEFAULT_AUDIO_ENCODING = "LINEAR16"
    DEFAULT_SPEAKING_RATE = "1"

    # API endpoint
    TTS_API_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"

    # Voice name mappings from OpenAI voices to Google Cloud voices
    # Users can pass either:
    # 1. OpenAI voice names (alloy, echo, fable, onyx, nova, shimmer) - will be mapped
    # 2. Google Cloud/Vertex AI voice names (en-US-Studio-O, en-US-Wavenet-D, etc.) - used directly
    VOICE_MAPPINGS = {
        "alloy": "en-US-Studio-O",
        "echo": "en-US-Studio-M",
        "fable": "en-GB-Studio-B",
        "onyx": "en-US-Wavenet-D",
        "nova": "en-US-Studio-O",
        "shimmer": "en-US-Wavenet-F",
    }

    # Response format mappings from OpenAI to Google Cloud audio encoding
    FORMAT_MAPPINGS = {
        "mp3": "MP3",
        "opus": "OGG_OPUS",
        "aac": "MP3",  # Google doesn't have AAC, use MP3
        "flac": "FLAC",
        "wav": "LINEAR16",
        "pcm": "LINEAR16",
    }

    def __init__(self) -> None:
        BaseTextToSpeechConfig.__init__(self)
        VertexBase.__init__(self)

    def _map_voice_to_vertex_format(
        self,
        voice: Union[str, dict] | None,
    ) -> tuple[str | None, dict | None]:
        """
        Map voice to Vertex AI format.

        Supports both:
        1. OpenAI voice names (alloy, echo, fable, onyx, nova, shimmer) - will be mapped
        2. Vertex AI voice names (en-US-Studio-O, en-US-Wavenet-D, etc.) - used directly
        3. Dict with languageCode and name - used as-is

        Returns:
            Tuple of (voice_str, voice_dict) where:
            - voice_str: Original string voice (for interface compatibility)
            - voice_dict: Vertex AI format dict with languageCode and name
        """
        if voice is None:
            return None, None

        if isinstance(voice, dict):
            # Already in Vertex AI format
            return None, voice

        # voice is a string
        voice_str = voice

        # Map OpenAI voice if it's a known OpenAI voice, otherwise use directly
        if voice in self.VOICE_MAPPINGS:
            mapped_voice_name = self.VOICE_MAPPINGS[voice]
        else:
            # Assume it's already a Vertex AI voice name
            mapped_voice_name = voice

        # Extract language code from voice name (e.g., "en-US-Studio-O" -> "en-US")
        parts = mapped_voice_name.split("-")
        if len(parts) >= 2:
            language_code = f"{parts[0]}-{parts[1]}"
        else:
            language_code = self.DEFAULT_LANGUAGE_CODE

        voice_dict = {
            "languageCode": language_code,
            "name": mapped_voice_name,
        }

        return voice_str, voice_dict

    def dispatch_text_to_speech(
        self,
        model: str,
        input: str,
        voice: Union[str, dict] | None,
        optional_params: dict,
        litellm_params_dict: dict,
        logging_obj: "LiteLLMLoggingObj",
        timeout: Union[float, httpx.Timeout],
        extra_headers: dict[str, Any] | None,
        base_llm_http_handler: Any,
        aspeech: bool,
        api_base: str | None,
        api_key: str | None,
        **kwargs: Any,
    ) -> Union[
        "HttpxBinaryResponseContent",
        Coroutine[Any, Any, "HttpxBinaryResponseContent"],
    ]:
        """
        Dispatch method to handle Vertex AI TTS requests

        This method encapsulates Vertex AI-specific credential resolution and parameter handling.
        Voice mapping is handled in map_openai_params (similar to Azure AVA pattern).

        Args:
            base_llm_http_handler: The BaseLLMHTTPHandler instance from main.py
        """
        # Resolve Vertex AI credentials using VertexBase helpers
        vertex_credentials = self.safe_get_vertex_ai_credentials(litellm_params_dict)
        vertex_project = self.safe_get_vertex_ai_project(litellm_params_dict)
        vertex_location = self.safe_get_vertex_ai_location(litellm_params_dict)

        # Convert voice to string if it's a dict (extract name)
        # Actual voice mapping happens in map_openai_params
        voice_str: str | None = None
        if isinstance(voice, str):
            voice_str = voice
        elif isinstance(voice, dict):
            # Extract voice name from dict if needed
            voice_str = voice.get("name") if voice else None

        # Store credentials in litellm_params for use in transform methods
        litellm_params_dict.update(
            {
                "vertex_credentials": vertex_credentials,
                "vertex_project": vertex_project,
                "vertex_location": vertex_location,
                "api_base": api_base,
            }
        )

        # Call the text_to_speech_handler
        response = base_llm_http_handler.text_to_speech_handler(
            model=model,
            input=input,
            voice=voice_str,
            text_to_speech_provider_config=self,
            text_to_speech_optional_params=optional_params,
            custom_llm_provider="vertex_ai",
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
        Vertex AI TTS supports these OpenAI parameters

        Note: Vertex AI also supports additional parameters like audioConfig
        which can be passed but are not part of the OpenAI spec
        """
        return ["voice", "response_format", "speed"]

    def map_openai_params(
        self,
        model: str,
        optional_params: dict,
        voice: str | dict | None = None,
        drop_params: bool = False,
        kwargs: dict = {},
    ) -> tuple[str | None, dict]:
        """
        Map OpenAI parameters to Vertex AI TTS parameters

        Voice handling (similar to Azure AVA):
        - If voice is an OpenAI voice name (alloy, echo, etc.), it maps to a Vertex AI voice
        - If voice is already a Vertex AI voice name (en-US-Studio-O, etc.), it's used directly
        - If voice is a dict with languageCode and name, it's used as-is

        Note: For Vertex AI, voice dict is stored in mapped_params["vertex_voice_dict"]
        because the base class interface expects voice to be a string.

        Returns:
            Tuple of (mapped_voice_str, mapped_params)
        """
        mapped_params: dict[str, Any] = {}

        ##########################################################
        # Map voice using helper
        ##########################################################
        mapped_voice_str, voice_dict = self._map_voice_to_vertex_format(voice)
        if voice_dict is not None:
            mapped_params["vertex_voice_dict"] = voice_dict

        # Map response format
        if "response_format" in optional_params:
            format_name = optional_params["response_format"]
            if format_name in self.FORMAT_MAPPINGS:
                mapped_params["audioEncoding"] = self.FORMAT_MAPPINGS[format_name]
            else:
                # Try to use it directly as Google Cloud format
                mapped_params["audioEncoding"] = format_name
        else:
            # Default to LINEAR16
            mapped_params["audioEncoding"] = self.DEFAULT_AUDIO_ENCODING

        # Map speed (OpenAI: 0.25-4.0, Vertex AI: speakingRate 0.25-4.0)
        if "speed" in optional_params:
            speed = optional_params["speed"]
            if speed is not None:
                mapped_params["speakingRate"] = str(speed)

        # Pass through Vertex AI-specific parameters from kwargs
        if "audioConfig" in kwargs:
            mapped_params["audioConfig"] = kwargs["audioConfig"]

        if "use_ssml" in kwargs:
            mapped_params["use_ssml"] = kwargs["use_ssml"]

        return mapped_voice_str, mapped_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        """
        Validate Vertex AI environment and set up authentication headers

        Note: Actual authentication is handled in transform_text_to_speech_request
        because Vertex AI requires OAuth2 token refresh
        """
        validated_headers = headers.copy()

        # Content-Type for JSON
        validated_headers["Content-Type"] = "application/json"
        validated_headers["charset"] = "UTF-8"

        return validated_headers

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        """
        Get the complete URL for Vertex AI TTS request

        Google Cloud TTS endpoint: https://texttospeech.googleapis.com/v1/text:synthesize
        """
        if api_base:
            return api_base

        return self.TTS_API_URL

    def _validate_vertex_input(
        self,
        input_data: VertexTextToSpeechInput,
        optional_params: dict,
    ) -> VertexTextToSpeechInput:
        """
        Validate and transform input for Vertex AI TTS

        Handles text vs SSML input detection and validation
        """
        # Remove None values
        if input_data.get("text") is None:
            input_data.pop("text", None)
        if input_data.get("ssml") is None:
            input_data.pop("ssml", None)

        # Check if use_ssml is set
        use_ssml = optional_params.get("use_ssml", False)

        if use_ssml:
            if "text" in input_data:
                input_data["ssml"] = input_data.pop("text")
            elif "ssml" not in input_data:
                raise ValueError("SSML input is required when use_ssml is True.")
        else:
            # LiteLLM will auto-detect if text is in ssml format
            # check if "text" is an ssml - in this case we should pass it as ssml instead of text
            if input_data:
                _text = input_data.get("text", None) or ""
                if "<speak>" in _text:
                    input_data["ssml"] = input_data.pop("text")

        if not input_data:
            raise ValueError("Either 'text' or 'ssml' must be provided.")
        if "text" in input_data and "ssml" in input_data:
            raise ValueError("Only one of 'text' or 'ssml' should be provided, not both.")

        return input_data

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
        Transform OpenAI TTS request to Vertex AI TTS format

        This method handles:
        1. Authentication with Vertex AI
        2. Building the request body
        3. Setting up headers

        Returns:
            TextToSpeechRequestData: Contains dict_body and headers
        """
        # Get Vertex AI credentials from litellm_params
        vertex_credentials: VERTEX_CREDENTIALS_TYPES | None = litellm_params.get("vertex_credentials")
        vertex_project: str | None = litellm_params.get("vertex_project")

        ####### Authenticate with Vertex AI ########
        _auth_header, vertex_project = self._ensure_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
            custom_llm_provider="vertex_ai_beta",
        )

        auth_header, _ = self._get_token_and_url(
            model="",
            auth_header=_auth_header,
            gemini_api_key=None,
            vertex_credentials=vertex_credentials,
            vertex_project=vertex_project,
            vertex_location=litellm_params.get("vertex_location"),
            stream=False,
            custom_llm_provider="vertex_ai_beta",
            api_base=litellm_params.get("api_base"),
        )

        # Set authentication headers
        headers["Authorization"] = f"Bearer {auth_header}"
        headers["x-goog-user-project"] = vertex_project

        ####### Build the request ################
        vertex_input = VertexTextToSpeechInput(text=input)
        vertex_input = self._validate_vertex_input(vertex_input, optional_params)

        # Build voice configuration
        # Check for voice dict stored in:
        # 1. litellm_params by dispatch method
        # 2. optional_params by map_openai_params
        voice_dict = litellm_params.get("vertex_voice_dict") or optional_params.get("vertex_voice_dict")
        if voice_dict is not None and isinstance(voice_dict, dict):
            vertex_voice = VertexTextToSpeechVoice(**voice_dict)
        elif voice is not None and isinstance(voice, str):
            # Handle string voice (shouldn't normally happen if dispatch was called)
            parts = voice.split("-")
            if len(parts) >= 2:
                language_code = f"{parts[0]}-{parts[1]}"
            else:
                language_code = self.DEFAULT_LANGUAGE_CODE
            vertex_voice = VertexTextToSpeechVoice(
                languageCode=language_code,
                name=voice,
            )
        else:
            # Use defaults
            vertex_voice = VertexTextToSpeechVoice(
                languageCode=self.DEFAULT_LANGUAGE_CODE,
                name=self.DEFAULT_VOICE_NAME,
            )

        # Build audio configuration
        audio_encoding = optional_params.get("audioEncoding", self.DEFAULT_AUDIO_ENCODING)
        speaking_rate = optional_params.get("speakingRate", self.DEFAULT_SPEAKING_RATE)

        # Check for full audioConfig in optional_params
        if "audioConfig" in optional_params:
            vertex_audio_config = VertexTextToSpeechAudioConfig(**optional_params["audioConfig"])
        else:
            vertex_audio_config = VertexTextToSpeechAudioConfig(
                audioEncoding=audio_encoding,
                speakingRate=speaking_rate,
            )

        request_body: dict[str, Any] = {
            "input": dict(vertex_input),
            "voice": dict(vertex_voice),
            "audioConfig": dict(vertex_audio_config),
        }

        return TextToSpeechRequestData(
            dict_body=request_body,
            headers=headers,
        )

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> "HttpxBinaryResponseContent":
        """
        Transform Vertex AI TTS response to standard format

        Vertex AI returns JSON with base64-encoded audio content.
        We decode it and return as HttpxBinaryResponseContent.
        """
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        # Parse JSON response
        _json_response = raw_response.json()

        # Get base64-encoded audio content
        response_content = _json_response.get("audioContent")
        if not response_content:
            raise ValueError("No audioContent in Vertex AI TTS response")

        # Decode base64 to get binary content
        binary_data = base64.b64decode(response_content)

        # Create an httpx.Response object with the binary data
        response = httpx.Response(
            status_code=200,
            content=binary_data,
        )

        # Initialize the HttpxBinaryResponseContent instance
        return HttpxBinaryResponseContent(response)


class VertexAILyriaTextToSpeechConfig(VertexAITextToSpeechConfig):
    @classmethod
    def is_lyria_model(cls, model: str) -> bool:
        return get_vertex_ai_lyria_model_info(model=model) is not None

    @staticmethod
    def _get_model_info(model: str) -> VertexAILyriaModelInfo:
        model_info = get_vertex_ai_lyria_model_info(model=model)
        if model_info is None:
            raise ValueError(f"Vertex AI model {model!r} does not declare a Lyria audio API")
        return model_info

    def get_supported_openai_params(self, model: str) -> list:
        return ["response_format"]

    def map_openai_params(
        self,
        model: str,
        optional_params: dict,
        voice: str | dict | None = None,
        drop_params: bool = False,
        kwargs: dict = {},
    ) -> tuple[str | None, dict]:
        mapped_params = dict(optional_params)
        base_model = model.removeprefix("vertex_ai/")
        model_info = self._get_model_info(model=model)
        unsupported_params = [param for param in ("speed", "instructions") if mapped_params.get(param) is not None]
        if unsupported_params:
            if drop_params or litellm.drop_params:
                for param in unsupported_params:
                    mapped_params.pop(param, None)
            else:
                raise UnsupportedParamsError(
                    status_code=400,
                    message=(
                        f"Vertex AI {base_model} does not support the OpenAI parameters: "
                        f"{', '.join(unsupported_params)}. To drop unsupported openai params "
                        "from the call, set `litellm.drop_params = True`"
                    ),
                )
        response_format = mapped_params.get("response_format")
        supported_formats = frozenset(model_info["supported_audio_formats"])
        if response_format is not None and response_format not in supported_formats:
            if drop_params or litellm.drop_params:
                mapped_params.pop("response_format", None)
            else:
                raise UnsupportedParamsError(
                    status_code=400,
                    message=(
                        f"Vertex AI {base_model} does not support response_format={response_format!r}. "
                        f"Supported values: {', '.join(sorted(supported_formats))}. "
                        "To drop unsupported openai params from the call, set `litellm.drop_params = True`"
                    ),
                )
        return voice if isinstance(voice, str) else None, mapped_params

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        base_model = model.removeprefix("vertex_ai/")
        model_info = self._get_model_info(model=model)
        project = self.safe_get_vertex_ai_project(litellm_params)
        if project is None:
            _, project = self._ensure_access_token(
                credentials=self.safe_get_vertex_ai_credentials(litellm_params),
                project_id=None,
                custom_llm_provider="vertex_ai",
            )
        if model_info["vertex_ai_audio_api"] == "lyria_interactions":
            from litellm.llms.vertex_ai.interactions.transformation import (
                VertexAIInteractionsConfig,
            )

            return VertexAIInteractionsConfig().get_complete_url(
                api_base=api_base,
                model=base_model,
                litellm_params={**litellm_params, "vertex_project": project},
            )
        location = self.safe_get_vertex_ai_location(litellm_params) or self.get_default_vertex_location()
        base_url = self.get_api_base(api_base=api_base, vertex_location=location).rstrip("/")
        return f"{base_url}/v1/projects/{project}/locations/{location}/publishers/google/models/{base_model}:predict"

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> TextToSpeechRequestData:
        access_token, project = self._ensure_access_token(
            credentials=self.safe_get_vertex_ai_credentials(litellm_params),
            project_id=self.safe_get_vertex_ai_project(litellm_params),
            custom_llm_provider="vertex_ai",
        )
        headers.update(
            {
                "Authorization": f"Bearer {access_token}",
                "x-goog-user-project": project,
                "Content-Type": "application/json",
            }
        )
        base_model = model.removeprefix("vertex_ai/")
        model_info = self._get_model_info(model=model)
        if model_info["vertex_ai_audio_api"] == "lyria_predict":
            request_body = {
                "instances": [{"prompt": input}],
                "parameters": {"sample_count": 1},
            }
        else:
            request_body = {"model": base_model, "input": input}
            if optional_params.get("response_format") == "wav":
                request_body["response_format"] = {
                    "type": "audio",
                    "mime_type": "audio/wav",
                }
        return TextToSpeechRequestData(dict_body=request_body, headers=headers)

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> "HttpxBinaryResponseContent":
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        response_json = raw_response.json()
        base_model = model.removeprefix("vertex_ai/")
        model_info = self._get_model_info(model=model)
        audio_data: str | None = None
        mime_type: str | None = None
        if model_info["vertex_ai_audio_api"] == "lyria_predict":
            predictions = response_json.get("predictions") or []
            if predictions:
                audio_data = predictions[0].get("audioContent") or predictions[0].get("bytesBase64Encoded")
                mime_type = predictions[0].get("mimeType")
        else:
            for step in response_json.get("steps") or response_json.get("outputs") or []:
                content_items = step.get("content") or [] if step.get("type") == "model_output" else [step]
                for content in content_items:
                    if content.get("type") == "audio" and content.get("data"):
                        audio_data = content["data"]
                        mime_type = content.get("mime_type")
        if audio_data is None:
            raise ValueError(f"No generated audio found in Vertex AI {base_model} response")
        default_format = model_info["supported_audio_formats"][0]
        mime_type = mime_type or {"mp3": "audio/mpeg", "wav": "audio/wav"}[default_format]
        response = HttpxBinaryResponseContent(
            httpx.Response(
                status_code=raw_response.status_code,
                content=base64.b64decode(audio_data),
                headers={"content-type": mime_type},
            )
        )
        response._hidden_params = {"audio_mime_type": mime_type}
        return response
