"""
Speechify text-to-speech via ``POST /v1/audio/speech``.

API reference: https://docs.speechify.ai/build/api-reference/v1/audio/speech?utm_source=github&utm_medium=docs&utm_campaign=litellm
"""

import base64
import binascii
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import BaseModel, ValidationError

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)
from litellm.secret_managers.main import get_secret_str

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import HttpxBinaryResponseContent


class SpeechifyTextToSpeechException(BaseLLMException):
    pass


class SpeechifySpeechResponse(BaseModel):
    audio_data: str
    audio_format: str


class SpeechifyTextToSpeechConfig(BaseTextToSpeechConfig):
    TTS_BASE_URL: Final[str] = "https://api.speechify.ai"
    DEFAULT_AUDIO_FORMAT: Final[str] = "mp3"
    OPENAI_TO_SPEECHIFY_FORMAT: Final[MappingProxyType[str, str]] = MappingProxyType({"opus": "ogg"})
    PASSTHROUGH_PARAMS: Final[tuple[str, ...]] = ("language", "output_format")
    AUDIO_CONTENT_TYPES: Final[MappingProxyType[str, str]] = MappingProxyType(
        {
            "mp3": "audio/mpeg",
            "wav": "audio/wav",
            "ogg": "audio/ogg",
            "aac": "audio/aac",
            "pcm": "audio/pcm",
            "l16": "audio/pcm",
            "ulaw": "audio/basic",
        }
    )
    DROPPED_RESPONSE_HEADERS: Final[frozenset[str]] = frozenset(
        {"content-encoding", "transfer-encoding", "content-length", "content-type"}
    )

    def get_supported_openai_params(self, model: str) -> list:  # mutable-ok: base class contract returns a plain list
        return ["voice", "response_format"]  # mutable-ok: base class contract returns a plain list

    def map_openai_params(
        self,
        model: str,
        optional_params: Mapping[str, object],
        voice: object = None,
        drop_params: bool = False,
        kwargs: Mapping[str, object] | None = None,
    ) -> tuple[str | None, dict]:  # mutable-ok: base class contract returns a plain dict
        response_format: Final = optional_params.get("response_format")
        audio_format: Final = (
            self.OPENAI_TO_SPEECHIFY_FORMAT.get(response_format, response_format)
            if isinstance(response_format, str)
            else self.DEFAULT_AUDIO_FORMAT
        )
        provider_params: Final = tuple((key, kwargs.get(key)) for key in self.PASSTHROUGH_PARAMS) if kwargs else ()
        mapped_params: Final = {  # mutable-ok: base class contract returns a plain dict
            key: value for key, value in (("audio_format", audio_format), *provider_params) if isinstance(value, str)
        }
        mapped_voice: Final = voice.strip() if isinstance(voice, str) and voice.strip() else None
        return mapped_voice, mapped_params

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: base class contract returns a plain dict
        resolved_key: Final = api_key or get_secret_str("SPEECHIFY_API_KEY")
        if resolved_key is None:
            raise SpeechifyTextToSpeechException(
                status_code=401,
                message="Speechify API key is required. Set SPEECHIFY_API_KEY or pass api_key.",
            )
        return {  # mutable-ok: base class contract returns a plain dict
            "Speechify-Caller": "litellm",
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
        configured_base: Final = (api_base or self.TTS_BASE_URL).rstrip("/")
        unversioned_base: Final = configured_base.removesuffix("/v1")
        return f"{unversioned_base}/v1/audio/speech"

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, str],
    ) -> TextToSpeechRequestData:
        request_data: Final[TextToSpeechRequestData] = {
            "dict_body": {
                "model": model,
                "input": input,
                **({"voice_id": voice} if voice else {}),
                **optional_params,
            },
            "headers": {"Content-Type": "application/json"},
        }
        return request_data

    def _content_type(self, audio_format: str) -> str:
        codec: Final = audio_format.partition(";")[0].strip().lower()
        return self.AUDIO_CONTENT_TYPES.get(codec, "application/octet-stream")

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> "HttpxBinaryResponseContent":
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        try:
            speech: Final = SpeechifySpeechResponse.model_validate_json(raw_response.content)
            audio_bytes: Final = base64.b64decode(speech.audio_data, validate=True)
        except (ValidationError, binascii.Error) as e:
            raise SpeechifyTextToSpeechException(
                status_code=500,
                message=f"Unexpected Speechify speech response: {e}",
                headers=raw_response.headers,
            )
        retained_headers: Final = tuple(
            (key, value)
            for key, value in raw_response.headers.items()
            if key.lower() not in self.DROPPED_RESPONSE_HEADERS
        )
        response_headers: Final = retained_headers + (
            ("content-length", str(len(audio_bytes))),
            ("content-type", self._content_type(speech.audio_format)),
        )
        return HttpxBinaryResponseContent(
            httpx.Response(
                status_code=200,
                headers=response_headers,
                content=audio_bytes,
                request=raw_response.request,
            )
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | httpx.Headers,  # mutable-ok: BaseLLMException takes a plain dict or httpx.Headers
    ) -> BaseLLMException:
        return SpeechifyTextToSpeechException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )
