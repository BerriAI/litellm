from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

import httpx
from pydantic import BaseModel, HttpUrl, ValidationError

import litellm
from litellm.litellm_core_utils.url_utils import async_safe_get, safe_get
from litellm.llms.alibaba_token_plan.common_utils import SPEECH_PATH, get_api_url, validate_headers
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.text_to_speech.transformation import BaseTextToSpeechConfig, TextToSpeechRequestData
from litellm.types.llms.openai import HttpxBinaryResponseContent

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_DEFAULT_VOICE: Final = "longanhuan_v3.6"


class _SpeechAudio(BaseModel, frozen=True):
    url: HttpUrl


class _SpeechOutput(BaseModel, frozen=True):
    audio: _SpeechAudio


class _SpeechResult(BaseModel, frozen=True):
    output: _SpeechOutput


class AlibabaTokenPlanTextToSpeechConfig(BaseTextToSpeechConfig):
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
        sample_rate: Final = (kwargs or {}).get("sample_rate")
        return _DEFAULT_VOICE if voice in (None, "alloy") else str(voice), {
            **({"format": optional_params["response_format"]} if "response_format" in optional_params else {}),
            **({"sample_rate": sample_rate} if sample_rate is not None else {}),
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
        return get_api_url(api_base, SPEECH_PATH)

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
                "input": {"text": input, "voice": voice or _DEFAULT_VOICE, "format": "mp3", **optional_params},
            }
        )

    def _audio_url(self, raw_response: httpx.Response) -> str:
        if not raw_response.is_success:
            raise self.get_error_class(raw_response.text, raw_response.status_code, raw_response.headers)
        try:
            return str(_SpeechResult.model_validate_json(raw_response.content).output.audio.url)
        except ValidationError:
            raise self.get_error_class(raw_response.text, 502, raw_response.headers) from None

    def _download_failed(self) -> BaseLLMException:
        return self.get_error_class("Alibaba Token Plan audio download failed", 502, {})

    def _downloaded(self, audio_response: httpx.Response) -> HttpxBinaryResponseContent:
        if not audio_response.is_success:
            raise self.get_error_class("Alibaba Token Plan audio download failed", audio_response.status_code, {})
        return HttpxBinaryResponseContent(audio_response)

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> HttpxBinaryResponseContent:
        audio_url: Final = self._audio_url(raw_response)
        try:
            return self._downloaded(safe_get(litellm.module_level_client, audio_url))
        except httpx.HTTPError:
            raise self._download_failed() from None

    async def async_transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> HttpxBinaryResponseContent:
        from litellm.llms.custom_httpx.http_handler import get_async_httpx_client

        audio_url: Final = self._audio_url(raw_response)
        client: Final = get_async_httpx_client(llm_provider=litellm.LlmProviders.ALIBABA_TOKEN_PLAN)
        try:
            return self._downloaded(await async_safe_get(client, audio_url))
        except httpx.HTTPError:
            raise self._download_failed() from None

    def get_error_class(
        self, error_message: str, status_code: int, headers: Mapping[str, str] | httpx.Headers
    ) -> BaseLLMException:
        return BaseLLMException(status_code=status_code, message=error_message, headers=dict(headers))
