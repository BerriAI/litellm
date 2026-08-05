import base64
import json
from typing import TYPE_CHECKING, Any

import httpx
from httpx import Headers

import litellm
from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import HttpxBinaryResponseContent

from ..common_utils import BytePlusError

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class BytePlusTextToSpeechConfig(BaseTextToSpeechConfig):
    """
    Configuration for BytePlus Text-to-Speech (HTTP Unidirectional Streaming).

    Reference:
    URL: https://voice.ap-southeast-1.bytepluses.com/api/v3/tts/unidirectional
    """

    DEFAULT_TTS_BASE_URL = "https://voice.ap-southeast-1.bytepluses.com/api/v3/tts/unidirectional"
    DEFAULT_APP_KEY = "aGjiRDfUWi"
    DEFAULT_SAMPLE_RATE = 24000
    DEFAULT_FORMAT = "mp3"

    def get_supported_openai_params(self, model: str) -> list:
        return ["voice", "response_format", "speed"]

    def map_openai_params(
        self,
        model: str,
        optional_params: dict,
        voice: str | dict | None = None,
        drop_params: bool = False,
        kwargs: dict[str, Any] | None = None,
    ) -> tuple[str | None, dict]:
        mapped_voice: str | None = None

        if isinstance(voice, str) and voice.strip():
            mapped_voice = voice.strip()
        elif isinstance(voice, dict):
            for key in ("speaker", "voice", "id", "name"):
                candidate = voice.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    mapped_voice = candidate.strip()
                    break

        if mapped_voice is None and kwargs:
            speaker_kwarg = kwargs.pop("speaker", None)
            if isinstance(speaker_kwarg, str) and speaker_kwarg.strip():
                mapped_voice = speaker_kwarg.strip()

        mapped_params = dict(optional_params) if optional_params else {}

        return mapped_voice, mapped_params

    VOICE_MAPPINGS = {
        "alloy": "en_female_stokie_uranus_bigtts",
        "echo": "en_male_adam_mars_bigtts",
        "fable": "en_female_skye_emo_v2_mars_bigtts",
        "onyx": "id_male_han_uranus_bigtts",
        "nova": "en_female_stokie_uranus_bigtts",
        "shimmer": "en_female_skye_emo_v2_mars_bigtts",
    }

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        api_key = (
            api_key
            or litellm.api_key
            or get_secret_str("BYTEPLUS_API_KEY")
            or get_secret_str("ARK_API_KEY")
        )
        app_id = get_secret_str("BYTEPLUS_TTS_APP_ID")
        access_key = get_secret_str("BYTEPLUS_TTS_ACCESS_KEY")
        app_key = get_secret_str("BYTEPLUS_TTS_APP_KEY") or self.DEFAULT_APP_KEY

        resource_id = model
        if resource_id.startswith("byteplus/"):
            resource_id = resource_id.replace("byteplus/", "", 1)

        req_headers = {
            "X-Api-Resource-Id": resource_id,
            "Content-Type": "application/json",
            "Connection": "keep-alive",
        }

        if api_key:
            clean_api_key = api_key.strip()
            req_headers["x-api-key"] = clean_api_key
        elif app_id and access_key:
            req_headers["X-Api-App-Id"] = app_id.strip()
            req_headers["X-Api-Access-Key"] = access_key.strip()
            req_headers["X-Api-App-Key"] = app_key.strip()
        else:
            raise ValueError(
                "BytePlus TTS requires authentication. Provide BYTEPLUS_API_KEY (or ARK_API_KEY) or both BYTEPLUS_TTS_APP_ID and BYTEPLUS_TTS_ACCESS_KEY."
            )

        headers.update(req_headers)
        return headers

    def get_error_class(self, error_message: str, status_code: int, headers: dict | Headers) -> BytePlusError:
        typed_headers: httpx.Headers = headers if isinstance(headers, httpx.Headers) else httpx.Headers(headers or {})
        return BytePlusError(status_code=status_code, message=error_message, headers=typed_headers)

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        tts_base_env = get_secret_str("BYTEPLUS_TTS_API_BASE")
        if tts_base_env:
            base_url = tts_base_env
        elif api_base and "voice" in api_base:
            base_url = api_base
        else:
            base_url = self.DEFAULT_TTS_BASE_URL

        base_url = base_url.rstrip("/")
        if base_url.endswith("/api/v3/tts/unidirectional"):
            return base_url
        if base_url.endswith("/api/v3"):
            return f"{base_url}/tts/unidirectional"
        if base_url.endswith("bytepluses.com"):
            return f"{base_url}/api/v3/tts/unidirectional"

        return base_url

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> TextToSpeechRequestData:
        params = dict(optional_params) if optional_params else {}
        extra_body = params.pop("extra_body", {}) or {}

        raw_speaker = voice or extra_body.get("speaker") or "en_female_stokie_uranus_bigtts"
        speaker = self.VOICE_MAPPINGS.get(raw_speaker.lower(), raw_speaker)

        audio_format = params.get("response_format") or extra_body.get("format") or self.DEFAULT_FORMAT
        sample_rate = extra_body.get("sample_rate") or self.DEFAULT_SAMPLE_RATE

        additions_dict: dict[str, Any] = {
            "disable_markdown_filter": True,
            "enable_language_detector": True,
            "enable_latex_tn": True,
            "disable_default_bit_rate": True,
            "max_length_to_filter_parenthesis": 0,
            "cache_config": {"text_type": 1, "use_cache": True},
        }

        user_additions = extra_body.get("additions")
        if isinstance(user_additions, str):
            try:
                parsed_user_additions = json.loads(user_additions)
                if isinstance(parsed_user_additions, dict):
                    additions_dict.update(parsed_user_additions)
            except Exception:
                pass
        elif isinstance(user_additions, dict):
            additions_dict.update(user_additions)

        user_id = extra_body.get("uid") or litellm_params.get("user") or "litellm-user"

        payload = {
            "user": {"uid": str(user_id)},
            "req_params": {
                "text": input,
                "speaker": speaker,
                "audio_params": {
                    "format": audio_format,
                    "sample_rate": sample_rate,
                },
                "additions": json.dumps(additions_dict),
            },
        }

        return TextToSpeechRequestData(
            dict_body=payload,
            headers={"Content-Type": "application/json"},
        )

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> HttpxBinaryResponseContent:
        """
        Processes BytePlus unidirectional streaming response.
        Extracts base64-encoded audio chunks from JSON lines and concatenates them into binary audio content.
        """
        if raw_response.status_code != 200:
            raise BytePlusError(
                status_code=raw_response.status_code,
                message=f"BytePlus TTS request failed: {raw_response.text}",
                headers=raw_response.headers,
            )

        audio_bytes = bytearray()
        chunk_count = 0

        lines = raw_response.text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except Exception:
                continue

            code = data.get("code", 0)

            if code == 0 and data.get("data"):
                try:
                    chunk_audio = base64.b64decode(data["data"])
                    audio_bytes.extend(chunk_audio)
                    chunk_count += 1
                except Exception:
                    pass
            elif code == 20000000:
                # Stream completed successfully
                break
            elif code > 0:
                msg = data.get("message") or f"BytePlus TTS error code {code}"
                raise BytePlusError(
                    status_code=400,
                    message=f"BytePlus TTS API Error: {msg}",
                    headers=raw_response.headers,
                )

        if not audio_bytes:
            raise BytePlusError(
                status_code=500,
                message="BytePlus TTS returned no audio data.",
                headers=raw_response.headers,
            )

        mock_http_response = httpx.Response(
            status_code=200,
            content=bytes(audio_bytes),
            headers={"content-type": "audio/mpeg"},
        )
        return HttpxBinaryResponseContent(mock_http_response)

    def dispatch_text_to_speech(
        self,
        model: str,
        input: str,
        voice: str | dict | None,
        optional_params: dict,
        litellm_params_dict: dict,
        logging_obj: "LiteLLMLoggingObj",
        timeout: float | httpx.Timeout,
        extra_headers: dict[str, Any] | None,
        base_llm_http_handler: Any,
        aspeech: bool,
        api_base: str | None,
        api_key: str | None,
        **kwargs: Any,
    ) -> Any:
        litellm_params_dict.update(
            {
                "api_key": api_key,
                "api_base": api_base,
            }
        )
        return base_llm_http_handler.text_to_speech_handler(
            model=model,
            input=input,
            voice=voice,
            text_to_speech_provider_config=self,
            text_to_speech_optional_params=optional_params,
            custom_llm_provider="byteplus",
            litellm_params=litellm_params_dict,
            logging_obj=logging_obj,
            timeout=timeout,
            extra_headers=extra_headers,
            client=None,
            _is_async=aspeech,
        )
