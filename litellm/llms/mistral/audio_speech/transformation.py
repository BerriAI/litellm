"""
Support for Mistral Voxtral text-to-speech via ``/v1/audio/speech``.

API reference: https://docs.mistral.ai/api/#tag/audio/operation/audio_speech_v1_audio_speech_post
"""

import base64
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)
from litellm.secret_managers.main import get_secret_str

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import HttpxBinaryResponseContent


class MistralTextToSpeechException(BaseLLMException):
    pass


class MistralTextToSpeechConfig(BaseTextToSpeechConfig):
    TTS_BASE_URL: Final[str] = "https://api.mistral.ai/v1"
    AUDIO_CONTENT_TYPES: Final[MappingProxyType[str, str]] = MappingProxyType(
        {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "pcm": "audio/pcm",
            "flac": "audio/flac",
            "opus": "audio/ogg",
        }
    )
    DROPPED_RESPONSE_HEADERS: Final[frozenset[str]] = frozenset(
        {"content-encoding", "transfer-encoding", "content-length", "content-type"}
    )
    OPENAI_VOICE_ALIASES: Final[MappingProxyType[str, str]] = MappingProxyType(
        {
            "alloy": "en_paul_neutral",
            "echo": "gb_oliver_neutral",
            "fable": "en_paul_cheerful",
            "onyx": "en_paul_confident",
            "nova": "gb_jane_sarcasm",
            "shimmer": "gb_jane_sarcasm",
        }
    )

    def get_supported_openai_params(self, model: str) -> list:  # mutable-ok: base class contract returns a plain list
        return ["voice", "response_format"]  # mutable-ok: base class contract returns a plain list

    def _map_openai_voice(self, voice_id: str) -> str:
        return self.OPENAI_VOICE_ALIASES.get(voice_id.lower(), voice_id)

    def _resolve_voice_id(self, voice: object) -> str | None:
        if isinstance(voice, str) and voice.strip():
            return self._map_openai_voice(voice.strip())
        if isinstance(voice, Mapping):
            candidates: Final = (voice.get(key) for key in ("voice_id", "id", "name"))
            resolved: Final = next(
                (candidate.strip() for candidate in candidates if isinstance(candidate, str) and candidate.strip()),
                None,
            )
            return self._map_openai_voice(resolved) if resolved else None
        return None

    def map_openai_params(
        self,
        model: str,
        optional_params: Mapping[str, object],
        voice: object = None,
        drop_params: bool = False,
        kwargs: Mapping[str, object] | None = None,
    ) -> tuple[str | None, dict]:  # mutable-ok: base class contract returns a plain dict
        response_format: Final = optional_params.get("response_format")
        ref_audio: Final = kwargs.get("ref_audio") if kwargs else None
        voice_id_kwarg: Final = kwargs.get("voice_id") if kwargs else None
        mapped_voice: Final = self._resolve_voice_id(voice) or self._resolve_voice_id(voice_id_kwarg)
        mapped_params: Final = {  # mutable-ok: base class contract returns a plain dict
            key: value
            for key, value in (("response_format", response_format), ("ref_audio", ref_audio))
            if isinstance(value, str)
        }
        return mapped_voice, mapped_params

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: base class contract returns a plain dict
        resolved_key: Final = api_key or get_secret_str("MISTRAL_API_KEY")
        if resolved_key is None:
            raise MistralTextToSpeechException(
                status_code=401,
                message="Mistral API key is required. Set MISTRAL_API_KEY or pass api_key.",
            )
        return {  # mutable-ok: base class contract returns a plain dict
            **headers,
            "Authorization": f"Bearer {resolved_key}",
            "Content-Type": "application/json",
        }

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: Mapping[str, object],
    ) -> str:
        base_url: Final = api_base or get_secret_str("MISTRAL_API_BASE") or self.TTS_BASE_URL
        return f"{base_url.rstrip('/')}/audio/speech"

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, str],
    ) -> TextToSpeechRequestData:
        response_format: Final = optional_params.get("response_format")
        ref_audio: Final = optional_params.get("ref_audio")
        request_data: Final[TextToSpeechRequestData] = {
            "dict_body": {
                "model": model,
                "input": input,
                **({"voice_id": voice} if voice else {}),
                **({"response_format": response_format} if isinstance(response_format, str) else {}),
                **({"ref_audio": ref_audio} if isinstance(ref_audio, str) else {}),
            },
            "headers": {"Content-Type": "application/json"},
        }
        return request_data

    def _requested_content_type(self, request: httpx.Request) -> str:
        request_body: Final = json.loads(request.content or b"{}")
        requested_format: Final = request_body.get("response_format")
        if not isinstance(requested_format, str):
            return "audio/mpeg"
        return self.AUDIO_CONTENT_TYPES.get(requested_format, "audio/mpeg")

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> "HttpxBinaryResponseContent":
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        try:
            response_json: Final = raw_response.json()
        except (json.JSONDecodeError, ValueError):
            raise MistralTextToSpeechException(
                status_code=raw_response.status_code,
                message=f"Non-JSON response from Mistral speech API: {raw_response.text[:500]}",
                headers=raw_response.headers,
            )
        audio_b64: Final = response_json.get("audio_data")
        if not isinstance(audio_b64, str) or not audio_b64:
            raise MistralTextToSpeechException(
                status_code=500,
                message=f"No audio_data in Mistral speech response. Response keys: {tuple(response_json.keys())}",
                headers=raw_response.headers,
            )
        try:
            audio_bytes: Final = base64.b64decode(audio_b64, validate=True)
        except ValueError:
            raise MistralTextToSpeechException(
                status_code=500,
                message="Invalid base64 audio_data in Mistral speech response.",
                headers=raw_response.headers,
            )
        retained_headers: Final = tuple(
            (key, value)
            for key, value in raw_response.headers.items()
            if key.lower() not in self.DROPPED_RESPONSE_HEADERS
        )
        response_headers: Final = retained_headers + (
            ("content-length", str(len(audio_bytes))),
            ("content-type", self._requested_content_type(raw_response.request)),
        )
        binary_response: Final = httpx.Response(
            status_code=200,
            headers=response_headers,
            content=audio_bytes,
            request=raw_response.request,
        )
        return HttpxBinaryResponseContent(binary_response)

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | httpx.Headers,  # mutable-ok: BaseLLMException takes a plain dict or httpx.Headers
    ) -> BaseLLMException:
        return MistralTextToSpeechException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
