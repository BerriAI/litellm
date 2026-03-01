"""
ModelsLab Text-to-Speech transformation for LiteLLM.

NOTE: ModelsLab uses key-in-body authentication. The MODELSLAB_API_KEY
will appear in the request body. Handle accordingly.

ModelsLab TTS API: https://docs.modelslab.com
"""
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import httpx

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)
from litellm.llms.custom_httpx.http_handler import HTTPHandler, _get_httpx_client
from litellm.secret_managers.main import get_secret_str

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import HttpxBinaryResponseContent
else:
    LiteLLMLoggingObj = Any
    HttpxBinaryResponseContent = Any

MODELSLAB_TTS_URL = "https://modelslab.com/api/v6/voice/text_to_speech"
MODELSLAB_TTS_FETCH_URL = "https://modelslab.com/api/v6/voice/fetch/{request_id}"
MODELSLAB_POLL_INTERVAL = 5
MODELSLAB_POLL_TIMEOUT = 300

# OpenAI voice → ModelsLab voice_id mappings
VOICE_MAPPINGS: Dict[str, int] = {
    "alloy": 1,    # neutral
    "echo": 2,     # male
    "fable": 3,    # warm
    "onyx": 4,     # deep male
    "nova": 5,     # female
    "shimmer": 6,  # clear female
}

LANGUAGE_MAPPINGS: Dict[str, str] = {
    "en": "english",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "pt": "portuguese",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
    "hi": "hindi",
    "ar": "arabic",
}


class ModelsLabTextToSpeechConfig(BaseTextToSpeechConfig):
    """
    Configuration for ModelsLab Text-to-Speech.

    ModelsLab TTS uses an async pattern:
    1. POST /api/v6/voice/text_to_speech → {status: success, output: "url"} or {status: processing, request_id: ...}
    2. If processing, poll POST /api/v6/voice/fetch/{request_id} with {key} body until done
    3. Download audio from output URL
    """

    def __init__(self):
        super().__init__()
        self._api_key: Optional[str] = None

    def get_supported_openai_params(self, model: str) -> list:
        return ["voice", "response_format", "speed"]

    def map_openai_params(
        self,
        model: str,
        optional_params: Dict,
        voice: Optional[Union[str, Dict]] = None,
        drop_params: bool = False,
        kwargs: Dict = {},
    ) -> Tuple[Optional[str], Dict]:
        mapped: Dict[str, Any] = {}

        # Resolve voice_id
        voice_id: Optional[int] = None
        if isinstance(voice, str) and voice.strip():
            voice_id = VOICE_MAPPINGS.get(voice.lower(), 1)
        elif isinstance(voice, dict):
            voice_id = voice.get("voice_id", 1)
        if voice_id:
            mapped["voice_id"] = voice_id

        # Map speed (ModelsLab accepts 0.5-2.0)
        if "speed" in optional_params:
            try:
                mapped["speed"] = float(optional_params["speed"])
            except (ValueError, TypeError):
                pass

        # Language from extra params
        if "language" in kwargs:
            lang = kwargs["language"]
            mapped["language"] = LANGUAGE_MAPPINGS.get(lang, lang)

        return voice, mapped

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """Key-in-body auth — only Content-Type goes in headers."""
        api_key = api_key or litellm.api_key or get_secret_str("MODELSLAB_API_KEY")

        if not api_key:
            raise ValueError(
                "ModelsLab API key is required. Set MODELSLAB_API_KEY or pass api_key."
            )

        self._api_key = api_key
        headers["Content-Type"] = "application/json"
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        if api_base:
            return api_base.rstrip("/")
        return MODELSLAB_TTS_URL

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: Optional[str],
        optional_params: Dict,
        litellm_params: Dict,
        headers: dict,
    ) -> TextToSpeechRequestData:
        body: Dict[str, Any] = {
            "key": self._api_key,
            "prompt": input,
            "language": optional_params.pop("language", "english"),
            "voice_id": optional_params.pop("voice_id", VOICE_MAPPINGS.get(voice or "", 1)),
            "speed": optional_params.pop("speed", 1.0),
        }
        # Pass through any remaining provider-specific params
        body.update(optional_params)
        return TextToSpeechRequestData(dict_body=body)

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> "HttpxBinaryResponseContent":
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        response_data = raw_response.json()
        status = response_data.get("status", "")
        request_id = str(response_data.get("request_id", ""))

        if status == "error":
            raise BaseLLMException(
                status_code=raw_response.status_code,
                message=response_data.get("message", "ModelsLab TTS failed"),
                headers=dict(raw_response.headers),
            )

        if status == "processing":
            response_data = self._poll_tts_sync(request_id)

        audio_url = response_data.get("output", "")
        if not audio_url:
            raise BaseLLMException(
                status_code=500,
                message="ModelsLab TTS returned no audio URL",
                headers={},
            )

        # Download the audio file
        client: HTTPHandler = _get_httpx_client()
        audio_response = client.get(audio_url)
        audio_response.raise_for_status()

        return HttpxBinaryResponseContent(audio_response)

    def _poll_tts_sync(
        self,
        request_id: str,
        timeout: int = MODELSLAB_POLL_TIMEOUT,
        interval: int = MODELSLAB_POLL_INTERVAL,
    ) -> Dict:
        """Poll the ModelsLab TTS fetch endpoint until done."""
        fetch_url = MODELSLAB_TTS_FETCH_URL.format(request_id=request_id)
        body = {"key": self._api_key}
        client: HTTPHandler = _get_httpx_client()
        deadline = time.time() + timeout

        while time.time() < deadline:
            time.sleep(interval)
            resp = client.post(fetch_url, json=body)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") in ("success", "error"):
                return data

        raise BaseLLMException(
            status_code=408,
            message=f"ModelsLab TTS timed out after {timeout}s (request_id={request_id})",
            headers={},
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Dict,
    ) -> BaseLLMException:
        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
