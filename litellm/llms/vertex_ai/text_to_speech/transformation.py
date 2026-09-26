"""
Vertex AI Text-to-Speech transformation

Maps OpenAI TTS spec to Google Cloud Text-to-Speech API
Reference: https://cloud.google.com/text-to-speech/docs/reference/rest/v1/text/synthesize
"""

import base64
import math
from collections.abc import Coroutine, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Final, TypeAlias, Union

import httpx

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.audio_utils.utils import (
    DEFAULT_SPEECH_MEDIA_TYPE,
    calculate_request_duration,
    speech_media_type_from_audio_bytes,
)
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
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
    VertexTextToSpeechSpeakerVoiceConfig,
    VertexTextToSpeechVoice,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
    from litellm.types.llms.openai import HttpxBinaryResponseContent
else:
    LiteLLMLoggingObj = Any
    HttpxBinaryResponseContent = Any

_LyriaVoice: TypeAlias = str | dict | None


def _fallback_gemini_tts_audio_duration(audio: bytes, encoding: str, sample_rate: int) -> float | None:
    if encoding == "PCM":
        return len(audio) / (2 * sample_rate) if sample_rate > 0 else None
    if encoding in ("ALAW", "MULAW") and not (audio[:4] == b"RIFF" and audio[8:12] == b"WAVE"):
        return len(audio) / sample_rate if sample_rate > 0 else None

    if encoding in ("LINEAR16", "ALAW", "MULAW") and audio[:4] == b"RIFF" and audio[8:12] == b"WAVE":
        format_offset: Final = audio.find(b"fmt ", 12)
        data_offset: Final = audio.find(b"data", 12)
        if format_offset >= 0 and data_offset >= 0 and format_offset + 20 <= len(audio):
            byte_rate: Final = int.from_bytes(audio[format_offset + 16 : format_offset + 20], "little")
            data_size: Final = int.from_bytes(audio[data_offset + 4 : data_offset + 8], "little")
            if byte_rate > 0 and data_size <= len(audio) - data_offset - 8:
                return data_size / byte_rate

    if encoding == "MP3" and speech_media_type_from_audio_bytes(audio) == "audio/mpeg":
        return len(audio) / 4000

    if encoding == "OGG_OPUS" and audio[:4] == b"OggS":
        opus_header: Final = audio.find(b"OpusHead")
        last_page: Final = audio.rfind(b"OggS")
        if opus_header >= 0 and opus_header + 12 <= len(audio) and last_page + 14 <= len(audio):
            pre_skip: Final = int.from_bytes(audio[opus_header + 10 : opus_header + 12], "little")
            granule: Final = int.from_bytes(audio[last_page + 6 : last_page + 14], "little")
            if pre_skip <= granule < 2**64 - 1:
                return (granule - pre_skip) / 48000

    return None


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
    FORMAT_MAPPINGS: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "mp3": "MP3",
            "opus": "OGG_OPUS",
            "aac": "MP3",  # Google doesn't have AAC, use MP3
            "flac": "FLAC",
            "wav": "LINEAR16",
            "pcm": "LINEAR16",
        }
    )
    GEMINI_FORMAT_MAPPINGS: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            **FORMAT_MAPPINGS,
            "alaw": "ALAW",
            "mulaw": "MULAW",
            "ogg_opus": "OGG_OPUS",
            "pcm": "LINEAR16",
            "pcm16": "LINEAR16",
            "linear16": "LINEAR16",
        }
    )

    def __init__(self) -> None:
        BaseTextToSpeechConfig.__init__(self)
        VertexBase.__init__(self)

    def _map_voice_to_vertex_format(
        self,
        voice: str | dict | None,
        model: str | None = None,
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

        if model is not None and self._is_gemini_tts_model(model):
            return self._map_gemini_tts_voice_to_vertex_format(model=model, voice=voice)

        if isinstance(voice, dict):
            # Already in Vertex AI format
            return None, voice

        # voice is a string
        voice_str: Final = voice

        # Map OpenAI voice if it's a known OpenAI voice, otherwise use directly
        if voice in self.VOICE_MAPPINGS:
            mapped_voice_name = self.VOICE_MAPPINGS[voice]
        else:
            # Assume it's already a Vertex AI voice name
            mapped_voice_name = voice

        # Extract language code from voice name (e.g., "en-US-Studio-O" -> "en-US")
        parts: Final = mapped_voice_name.split("-")
        if len(parts) >= 2:
            language_code = f"{parts[0]}-{parts[1]}"
        else:
            language_code = self.DEFAULT_LANGUAGE_CODE

        voice_dict: Final = {
            "languageCode": language_code,
            "name": mapped_voice_name,
        }

        return voice_str, voice_dict

    @staticmethod
    def _is_gemini_tts_model(model: str) -> bool:
        from litellm.utils import is_gemini_tts_model

        return is_gemini_tts_model(model, custom_llm_provider="vertex_ai")

    @staticmethod
    def _get_str_value(
        source: Mapping[str, object],
        *keys: str,
    ) -> str | None:
        for key in keys:
            value = source.get(key)
            if isinstance(value, str):
                return value
        return None

    @staticmethod
    def _get_dict_value(
        source: Mapping[str, object],
        *keys: str,
    ) -> dict[str, object] | None:  # mutable-ok: nested provider payloads remain concrete dictionaries
        for key in keys:
            value = source.get(key)
            if isinstance(value, dict):
                return value
        return None

    def _extract_gemini_tts_speaker_configs(
        self,
        voice: Mapping[str, object],
    ) -> list[
        VertexTextToSpeechSpeakerVoiceConfig
    ]:  # mutable-ok: provider request serialization requires a concrete list
        speech_config: Final = self._get_dict_value(voice, "speechConfig", "speech_config") or voice
        multi_speaker_config: Final = self._get_dict_value(
            speech_config,
            "multiSpeakerVoiceConfig",
            "multi_speaker_voice_config",
        )
        if multi_speaker_config is None:
            return []  # mutable-ok: provider request serialization requires a concrete empty list
        raw_speaker_configs: Final = multi_speaker_config.get(
            "speakerVoiceConfigs",
            multi_speaker_config.get(
                "speaker_voice_configs",
                [],  # mutable-ok: missing speaker configuration uses a concrete empty-list sentinel
            ),
        )
        if not isinstance(raw_speaker_configs, list):
            return []  # mutable-ok: malformed speaker configuration produces a concrete empty list
        speaker_configs: Final[  # mutable-ok: validated payloads are accumulated for serialization
            list[VertexTextToSpeechSpeakerVoiceConfig]
        ] = []
        for raw_config in raw_speaker_configs:
            if not isinstance(raw_config, dict):
                continue
            speaker_alias = self._get_str_value(
                raw_config,
                "speakerAlias",
                "speaker_alias",
                "speaker",
            )
            speaker_id = self._get_str_value(raw_config, "speakerId", "speaker_id")
            if speaker_id is None:
                voice_config = self._get_dict_value(raw_config, "voiceConfig", "voice_config")
                if voice_config is not None:
                    prebuilt_voice_config = self._get_dict_value(
                        voice_config,
                        "prebuiltVoiceConfig",
                        "prebuilt_voice_config",
                    )
                    if prebuilt_voice_config is not None:
                        speaker_id = self._get_str_value(
                            prebuilt_voice_config,
                            "voiceName",
                            "voice_name",
                        )
            if speaker_alias is not None and speaker_id is not None:
                speaker_configs.append(
                    {  # mutable-ok: provider request serialization requires a concrete dict
                        "speakerAlias": speaker_alias,
                        "speakerId": speaker_id,
                    }
                )
        return speaker_configs

    def _extract_gemini_tts_voice_name(
        self,
        voice: Mapping[str, object],
    ) -> str | None:
        voice_name: Final = self._get_str_value(voice, "name", "voiceName", "voice_name", "voice")
        if voice_name is not None:
            return voice_name
        speech_config: Final = self._get_dict_value(voice, "speechConfig", "speech_config") or voice
        nested_voice_name: Final = self._get_str_value(speech_config, "name", "voiceName", "voice_name", "voice")
        if nested_voice_name is not None:
            return nested_voice_name
        voice_config: Final = self._get_dict_value(speech_config, "voiceConfig", "voice_config")
        if voice_config is None:
            return None
        prebuilt_voice_config: Final = self._get_dict_value(
            voice_config,
            "prebuiltVoiceConfig",
            "prebuilt_voice_config",
        )
        if prebuilt_voice_config is None:
            return None
        return self._get_str_value(prebuilt_voice_config, "voiceName", "voice_name")

    def _map_gemini_tts_voice_to_vertex_format(
        self,
        model: str,
        voice: str | Mapping[str, object],
    ) -> tuple[str | None, dict[str, object]]:  # mutable-ok: provider request serialization requires a concrete dict
        if isinstance(voice, str):
            return voice, {  # mutable-ok: provider request serialization requires a concrete dict
                "languageCode": self.DEFAULT_LANGUAGE_CODE,
                "modelName": model,
                "name": voice,
            }

        speech_config: Final = self._get_dict_value(voice, "speechConfig", "speech_config") or voice
        language_code: Final = (
            self._get_str_value(voice, "languageCode", "language_code")
            or self._get_str_value(speech_config, "languageCode", "language_code")
            or self.DEFAULT_LANGUAGE_CODE
        )
        model_name: Final = model
        speaker_configs: Final = self._extract_gemini_tts_speaker_configs(voice)
        if speaker_configs:
            return None, {  # mutable-ok: provider request serialization requires a concrete dict
                "languageCode": language_code,
                "modelName": model_name,
                "multiSpeakerVoiceConfig": {  # mutable-ok: nested provider payload is serialized as a dict
                    "speakerVoiceConfigs": speaker_configs,
                },
            }
        voice_name: Final = self._extract_gemini_tts_voice_name(voice)
        if voice_name is not None:
            return None, {  # mutable-ok: provider request serialization requires a concrete dict
                "languageCode": language_code,
                "modelName": model_name,
                "name": voice_name,
            }
        return None, {  # mutable-ok: provider request serialization requires a concrete dict
            **voice,
            "languageCode": language_code,
            "modelName": model_name,
        }

    @staticmethod
    def _dispatch_voice_name(voice: str | Mapping[str, object] | None) -> str | None:
        if isinstance(voice, str):
            return voice
        if not isinstance(voice, Mapping):
            return None
        name: Final = voice.get("name")
        return name if isinstance(name, str) else None

    def dispatch_text_to_speech(
        self,
        model: str,
        input: str,
        voice: str | dict | None,
        optional_params: dict[str, object],
        litellm_params_dict: dict[str, object],
        logging_obj: "LiteLLMLoggingObj",
        timeout: float | httpx.Timeout,
        extra_headers: dict[str, object] | None,
        base_llm_http_handler: "BaseLLMHTTPHandler",
        aspeech: bool,
        api_base: str | None,
        api_key: str | None,
        **kwargs: object,
    ) -> Union[
        "HttpxBinaryResponseContent",
        Coroutine[object, object, "HttpxBinaryResponseContent"],
    ]:
        """
        Dispatch method to handle Vertex AI TTS requests

        This method encapsulates Vertex AI-specific credential resolution and parameter handling.
        Voice mapping is handled in map_openai_params (similar to Azure AVA pattern).

        Args:
            base_llm_http_handler: The BaseLLMHTTPHandler instance from main.py
        """
        # Resolve Vertex AI credentials using VertexBase helpers
        vertex_credentials: Final = self.safe_get_vertex_ai_credentials(litellm_params_dict)
        vertex_project: Final = self.safe_get_vertex_ai_project(litellm_params_dict)
        vertex_location: Final = self.safe_get_vertex_ai_location(litellm_params_dict)

        mapped_voice, mapped_params = (
            (self._dispatch_voice_name(voice), optional_params)
            if "audioEncoding" in optional_params
            else self.map_openai_params(
                model=model,
                voice=voice,
                optional_params=optional_params,
                kwargs=kwargs,
            )
        )

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
        response: Final = base_llm_http_handler.text_to_speech_handler(
            model=model,
            input=input,
            voice=mapped_voice,
            text_to_speech_provider_config=self,
            text_to_speech_optional_params=mapped_params,
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
        mapped_params: Final[dict[str, object]] = {}

        ##########################################################
        # Map voice using helper
        ##########################################################
        mapped_voice_str, voice_dict = self._map_voice_to_vertex_format(
            voice=voice,
            model=model,
        )
        if voice_dict is not None:
            mapped_params["vertex_voice_dict"] = voice_dict

        # Map response format
        if "response_format" in optional_params:
            format_name: Final = optional_params["response_format"]
            format_mappings: Final = (
                self.GEMINI_FORMAT_MAPPINGS if self._is_gemini_tts_model(model) else self.FORMAT_MAPPINGS
            )
            if format_name in format_mappings:
                mapped_params["audioEncoding"] = format_mappings[format_name]
            else:
                # Try to use it directly as Google Cloud format
                mapped_params["audioEncoding"] = format_name
        else:
            # Default to LINEAR16
            mapped_params["audioEncoding"] = self.DEFAULT_AUDIO_ENCODING

        # Map speed (OpenAI: 0.25-4.0, Vertex AI: speakingRate 0.25-4.0)
        if "speed" in optional_params:
            speed: Final = optional_params["speed"]
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
        validated_headers: Final = headers.copy()

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
        use_ssml: Final = optional_params.get("use_ssml", False)

        if use_ssml:
            if "text" in input_data:
                input_data["ssml"] = input_data.pop("text")
            elif "ssml" not in input_data:
                raise ValueError("SSML input is required when use_ssml is True.")
        else:
            # LiteLLM will auto-detect if text is in ssml format
            # check if "text" is an ssml - in this case we should pass it as ssml instead of text
            if input_data:
                _text: Final = input_data.get("text", None) or ""
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
        vertex_credentials: Final[VERTEX_CREDENTIALS_TYPES | None] = litellm_params.get("vertex_credentials")
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
        voice_dict: Final = litellm_params.get("vertex_voice_dict") or optional_params.get("vertex_voice_dict")
        if voice_dict is not None and isinstance(voice_dict, dict):
            vertex_voice = VertexTextToSpeechVoice(**voice_dict)
        elif voice is not None:
            # Handle string voice (shouldn't normally happen if dispatch was called)
            parts: Final = voice.split("-")
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
        audio_encoding: Final = optional_params.get("audioEncoding", self.DEFAULT_AUDIO_ENCODING)
        speaking_rate: Final = optional_params.get("speakingRate", self.DEFAULT_SPEAKING_RATE)

        # Check for full audioConfig in optional_params
        if "audioConfig" in optional_params:
            vertex_audio_config = VertexTextToSpeechAudioConfig(**optional_params["audioConfig"])
        else:
            vertex_audio_config = VertexTextToSpeechAudioConfig(
                audioEncoding=audio_encoding,
                speakingRate=speaking_rate,
            )

        request_body: Final[dict[str, object]] = {
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
        _json_response: Final = raw_response.json()

        # Get base64-encoded audio content
        response_content: Final = _json_response.get("audioContent")
        if not response_content:
            raise ValueError("No audioContent in Vertex AI TTS response")

        binary_data: Final = base64.b64decode(response_content)
        media_type: Final = speech_media_type_from_audio_bytes(binary_data)
        response: Final = httpx.Response(
            status_code=200,
            headers=None if media_type is None else MappingProxyType({"content-type": media_type}),
            content=binary_data,
        )

        binary_response: Final = HttpxBinaryResponseContent(response)
        if self._is_gemini_tts_model(model):
            from litellm.types.utils import CompletionTokensDetailsWrapper, Usage

            request_body: Final = logging_obj.model_call_details["additional_args"]["complete_input_dict"]["dict_body"]
            input_data: Final = request_body["input"]
            audio_config: Final = request_body["audioConfig"]
            container_duration: Final = calculate_request_duration(binary_data)
            sample_rate: Final = audio_config.get("sampleRateHertz") or 24000
            duration: Final = (
                container_duration
                if container_duration is not None
                else _fallback_gemini_tts_audio_duration(binary_data, audio_config["audioEncoding"], sample_rate)
            )
            if duration is None:
                raise ValueError("Cannot determine Gemini TTS output duration for cost calculation")
            input_text: Final = " ".join(
                value for value in (input_data.get("text"), input_data.get("ssml"), input_data.get("prompt")) if value
            )
            prompt_tokens: Final = litellm.token_counter(model=model, text=input_text)
            audio_tokens: Final = math.ceil(duration * 25)
            binary_response.usage = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=audio_tokens,
                total_tokens=prompt_tokens + audio_tokens,
                completion_tokens_details=CompletionTokensDetailsWrapper(audio_tokens=audio_tokens),
            )
        return binary_response


class VertexAILyriaTextToSpeechConfig(VertexAITextToSpeechConfig):
    @classmethod
    def is_lyria_model(cls, model: str) -> bool:
        return get_vertex_ai_lyria_model_info(model=model) is not None

    @staticmethod
    def _get_model_info(model: str) -> VertexAILyriaModelInfo:
        model_info: Final = get_vertex_ai_lyria_model_info(model=model)
        if model_info is None:
            raise ValueError(f"Vertex AI model {model!r} does not declare a Lyria audio API")
        return model_info

    def get_supported_openai_params(
        self, model: str
    ) -> list:  # mutable-ok: inherited provider interface returns a concrete parameter list
        return [  # mutable-ok: inherited provider interface requires a concrete parameter list
            "response_format"
        ]

    def map_openai_params(
        self,
        model: str,
        optional_params: dict,  # mutable-ok: inherited provider interface accepts a concrete parameter dictionary
        voice: _LyriaVoice = None,
        drop_params: bool = False,
        kwargs: dict | None = None,  # mutable-ok: inherited provider interface accepts a concrete keyword dictionary
    ) -> tuple[str | None, dict]:  # mutable-ok: inherited provider interface returns concrete mapped parameters
        mapped_params: Final = dict(  # mutable-ok: mapping drops unsupported parameters before provider dispatch
            optional_params
        )
        base_model: Final = model.removeprefix("vertex_ai/")
        model_info: Final = self._get_model_info(model=model)
        unsupported_params: Final = tuple(
            param for param in ("speed", "instructions") if mapped_params.get(param) is not None
        )
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
        response_format: Final = mapped_params.get("response_format")
        supported_formats: Final = frozenset(model_info["supported_audio_formats"])
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
        litellm_params: dict,  # mutable-ok: inherited provider interface accepts concrete LiteLLM parameters
    ) -> str:
        base_model: Final = model.removeprefix("vertex_ai/")
        model_info: Final = self._get_model_info(model=model)
        configured_project: Final = self.safe_get_vertex_ai_project(litellm_params)
        project: Final = (
            self._ensure_access_token(
                credentials=self.safe_get_vertex_ai_credentials(litellm_params),
                project_id=None,
                custom_llm_provider="vertex_ai",
            )[1]
            if configured_project is None
            else configured_project
        )
        if model_info["vertex_ai_audio_api"] == "lyria_interactions":
            from litellm.llms.vertex_ai.interactions.transformation import (
                VertexAIInteractionsConfig,
            )

            def mint_access_token(
                _credentials: VERTEX_CREDENTIALS_TYPES | None,
                project_id: str | None,
            ) -> tuple[str, str]:
                return "", project_id or project

            return VertexAIInteractionsConfig(mint_access_token=mint_access_token).get_complete_url(
                api_base=api_base,
                model=base_model,
                litellm_params={  # mutable-ok: interactions dispatch expects a concrete parameter dictionary
                    **litellm_params,
                    "vertex_project": project,
                    "vertex_location": "global",
                },
            )
        location: Final = self.safe_get_vertex_ai_location(litellm_params) or self.get_default_vertex_location()
        base_url: Final = self.get_api_base(api_base=api_base, vertex_location=location).rstrip("/")
        encoded_project: Final = encode_url_path_segment(project, field_name="project")
        encoded_location: Final = encode_url_path_segment(location, field_name="location")
        encoded_model: Final = encode_url_path_segment(base_model, field_name="model")
        return (
            f"{base_url}/v1/projects/{encoded_project}/locations/{encoded_location}"
            f"/publishers/google/models/{encoded_model}:predict"
        )

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: dict,  # mutable-ok: inherited provider interface accepts concrete mapped parameters
        litellm_params: dict,  # mutable-ok: inherited provider interface accepts concrete LiteLLM parameters
        headers: dict,  # mutable-ok: inherited provider interface accepts and updates concrete HTTP headers
    ) -> TextToSpeechRequestData:
        access_token, project = self._ensure_access_token(
            credentials=self.safe_get_vertex_ai_credentials(litellm_params),
            project_id=self.safe_get_vertex_ai_project(litellm_params),
            custom_llm_provider="vertex_ai",
        )
        headers.update(
            {  # mutable-ok: HTTP dispatch requires a concrete header dictionary
                "Authorization": f"Bearer {access_token}",
                "x-goog-user-project": project,
                "Content-Type": "application/json",
            }
        )
        base_model: Final = model.removeprefix("vertex_ai/")
        model_info: Final = self._get_model_info(model=model)
        request_body: Final[dict[str, object]] = (  # mutable-ok: HTTP dispatch requires a concrete provider payload
            {  # mutable-ok: predict dispatch requires a concrete provider request dictionary
                "instances": [  # mutable-ok: predict dispatch requires a concrete instances list
                    {"prompt": input}  # mutable-ok: predict dispatch requires a concrete instance dictionary
                ],
                "parameters": {  # mutable-ok: predict dispatch requires a concrete parameters dictionary
                    "sample_count": 1
                },
            }
            if model_info["vertex_ai_audio_api"] == "lyria_predict"
            else {  # mutable-ok: interactions dispatch requires a concrete provider request dictionary
                "model": base_model,
                "input": input,
                **(
                    {  # mutable-ok: interactions dispatch requires a nested response-format dictionary
                        "response_format": {  # mutable-ok: interactions response format is a concrete provider payload
                            "type": "audio",
                            "mime_type": "audio/wav",
                        }
                    }
                    if optional_params.get("response_format") == "wav"
                    else {}  # mutable-ok: no response override is merged for non-WAV output
                ),
            }
        )
        return TextToSpeechRequestData(dict_body=request_body, headers=headers)

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> "HttpxBinaryResponseContent":
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        response_json: Final = raw_response.json()
        base_model: Final = model.removeprefix("vertex_ai/")
        model_info: Final = self._get_model_info(model=model)
        audio_data: str | None = None  # rebind-ok: response parsing discovers audio data in provider-specific shapes
        mime_type: str | None = None  # rebind-ok: response parsing discovers the MIME type beside the audio payload
        if model_info["vertex_ai_audio_api"] == "lyria_predict":
            predictions: Final = response_json.get("predictions") or ()
            if predictions:
                audio_data = predictions[0].get("audioContent") or predictions[0].get("bytesBase64Encoded")
                mime_type = predictions[0].get("mimeType")  # rebind-ok: predict response supplies its audio MIME type
        else:
            for step in response_json.get("steps") or response_json.get("outputs") or ():
                content_items = step.get("content") or () if step.get("type") == "model_output" else (step,)
                for content in content_items:
                    if content.get("type") == "audio" and content.get("data"):
                        audio_data = content["data"]
                        mime_type = content.get("mime_type")
        if audio_data is None:
            raise ValueError(f"No generated audio found in Vertex AI {base_model} response")
        binary_data: Final = base64.b64decode(audio_data)
        media_type: Final = mime_type or speech_media_type_from_audio_bytes(binary_data) or DEFAULT_SPEECH_MEDIA_TYPE
        return HttpxBinaryResponseContent(
            httpx.Response(
                status_code=raw_response.status_code,
                content=binary_data,
                headers=MappingProxyType({"content-type": media_type}),
            )
        )
