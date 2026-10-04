from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import BaseModel, HttpUrl, ValidationError

import litellm
from litellm.litellm_core_utils.url_utils import async_safe_get, safe_get
from litellm.llms.alibaba_token_plan.common_utils import SPEECH_ENDPOINT, get_native_api_url, validate_headers
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.text_to_speech.transformation import BaseTextToSpeechConfig, TextToSpeechRequestData
from litellm.types.llms.openai import HttpxBinaryResponseContent

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


_DEFAULT_VOICE: Final = "longanhuan_v3.6"


class _SpeechAudio(BaseModel, frozen=True):
    data: str | None = None
    url: HttpUrl | None = None


class _SpeechOutput(BaseModel, frozen=True):
    audio: _SpeechAudio


class _SpeechResult(BaseModel, frozen=True):
    output: _SpeechOutput | None = None
    code: str | None = None
    message: str | None = None


class AlibabaTokenPlanTextToSpeechConfig(BaseTextToSpeechConfig):
    def _get_audio(self, raw_response: httpx.Response) -> _SpeechAudio | None:
        if not raw_response.is_success:
            raise self.get_error_class(raw_response.text, raw_response.status_code, raw_response.headers)
        content_type: Final = dict(raw_response.headers).get("content-type", "")
        if "json" not in content_type.lower():
            return None
        try:
            result: Final = _SpeechResult.model_validate_json(raw_response.content)
        except ValidationError:
            raise self.get_error_class(
                "Invalid Alibaba Token Plan speech response", 502, raw_response.headers
            ) from None
        if result.code or result.output is None:
            raise self.get_error_class(
                result.message or result.code or "Alibaba Token Plan returned no audio output",
                502,
                raw_response.headers,
            )
        if not result.output.audio.data and not result.output.audio.url:
            raise self.get_error_class("Alibaba Token Plan returned no audio", 502, raw_response.headers)
        return result.output.audio

    def get_error_class(
        self, error_message: str, status_code: int, headers: Mapping[str, str] | httpx.Headers
    ) -> BaseLLMException:
        return BaseLLMException(status_code=status_code, message=error_message, headers=dict(headers))

    def get_supported_openai_params(
        self, model: str
    ) -> list[str]:  # mutable-ok: provider interface requires a mutable return value
        return ["voice", "response_format"]

    def map_openai_params(
        self,
        model: str,
        optional_params: Mapping[str, object],
        voice: str | Mapping[str, object] | None = None,
        drop_params: bool = False,
        kwargs: Mapping[str, object] | None = None,
    ) -> tuple[str | None, dict[str, object]]:  # mutable-ok: provider interface requires a mutable return value
        unsupported: Final = tuple(key for key in optional_params if key not in self.get_supported_openai_params(model))
        if unsupported and not (drop_params or litellm.drop_params):
            raise litellm.UnsupportedParamsError(
                message=f"Alibaba Token Plan speech does not support: {', '.join(unsupported)}",
                model=model,
                llm_provider="alibaba_token_plan",
            )
        if voice is not None and not isinstance(voice, str):
            raise ValueError("Alibaba Token Plan voice must be a voice name string")
        native_params: Final = {key: value for key, value in (kwargs or {}).items() if key in ("sample_rate", "format")}
        return _DEFAULT_VOICE if voice == "alloy" else voice, {
            **native_params,
            **({"format": optional_params["response_format"]} if "response_format" in optional_params else {}),
        }

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: provider interface requires a mutable return value
        return validate_headers(headers, api_key)

    def get_complete_url(self, model: str, api_base: str | None, litellm_params: Mapping[str, object]) -> str:
        return get_native_api_url(api_base, SPEECH_ENDPOINT)

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, str],
    ) -> TextToSpeechRequestData:
        return TextToSpeechRequestData(
            dict_body={
                "model": model,
                "input": {
                    "text": input,
                    "voice": voice or _DEFAULT_VOICE,
                    "format": "mp3",
                    "sample_rate": 24000,
                    **optional_params,
                },
            }
        )

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> HttpxBinaryResponseContent:
        audio: Final = self._get_audio(raw_response)
        if audio is None:
            return HttpxBinaryResponseContent(raw_response)
        inline_response: Final = self._transform_inline_audio(audio, raw_response)
        if inline_response is not None:
            return inline_response
        if audio.url is None:
            raise self.get_error_class("Alibaba Token Plan returned no audio", 502, raw_response.headers)

        try:
            audio_response: Final = safe_get(litellm.module_level_client, str(audio.url))
        except httpx.HTTPError:
            raise self.get_error_class("Alibaba Token Plan audio download failed", 502, {}) from None
        if not audio_response.is_success:
            raise self.get_error_class(
                "Alibaba Token Plan audio download failed",
                audio_response.status_code,
                {},
            )
        return HttpxBinaryResponseContent(audio_response)

    async def async_transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> HttpxBinaryResponseContent:
        audio: Final = self._get_audio(raw_response)
        if audio is None:
            return HttpxBinaryResponseContent(raw_response)
        inline_response: Final = self._transform_inline_audio(audio, raw_response)
        if inline_response is not None:
            return inline_response
        if audio.url is None:
            raise self.get_error_class("Alibaba Token Plan returned no audio", 502, raw_response.headers)

        from litellm.llms.custom_httpx.http_handler import get_async_httpx_client

        try:
            audio_response: Final = await async_safe_get(
                get_async_httpx_client(llm_provider=litellm.LlmProviders.ALIBABA_TOKEN_PLAN),
                str(audio.url),
            )
        except httpx.HTTPError:
            raise self.get_error_class("Alibaba Token Plan audio download failed", 502, {}) from None
        if not audio_response.is_success:
            raise self.get_error_class(
                "Alibaba Token Plan audio download failed",
                audio_response.status_code,
                {},
            )
        return HttpxBinaryResponseContent(audio_response)

    def _transform_inline_audio(
        self,
        audio: _SpeechAudio,
        raw_response: httpx.Response,
    ) -> HttpxBinaryResponseContent | None:
        if not audio.data:
            return None
        try:
            audio_bytes: Final = base64.b64decode(audio.data, validate=True)
        except (binascii.Error, ValueError) as error:
            raise self.get_error_class(f"Invalid Alibaba Token Plan audio data: {error}", 502, raw_response.headers)
        return HttpxBinaryResponseContent(
            httpx.Response(
                status_code=200,
                content=audio_bytes,
                headers={"Content-Type": "application/octet-stream"},
                request=raw_response.request,
            )
        )
