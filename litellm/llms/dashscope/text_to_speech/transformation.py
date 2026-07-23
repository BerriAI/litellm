"""
DashScope Text-to-Speech transformation

Maps the OpenAI TTS spec to the DashScope multimodal-generation API used by the
Qwen-TTS model family (including qwen3-tts-vc voice cloning).

API endpoint: POST https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation

Request format:
{
    "model": "qwen3-tts-vc",
    "input": {"text": "<text>", "voice": "<voice>"}
}

Non-streaming response returns an audio file URL (valid for 24h):
{
    "output": {"audio": {"url": "<url>"}},
    "usage": {...}
}
Reference: https://www.alibabacloud.com/help/en/model-studio/non-realtime-tts-user-guide
"""

from typing import TYPE_CHECKING, Any

import httpx
from httpx import Headers

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)
from litellm.llms.dashscope.common_utils import DashScopeError
from litellm.secret_managers.main import get_secret_str

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import HttpxBinaryResponseContent
else:
    LiteLLMLoggingObj = Any
    HttpxBinaryResponseContent = Any

DEFAULT_API_BASE = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

UNSUPPORTED_OPENAI_PARAMS = ("response_format", "speed", "instructions")


class DashScopeTextToSpeechConfig(BaseTextToSpeechConfig):
    """
    Configuration for DashScope (Qwen) text-to-speech.
    """

    def get_supported_openai_params(self, model: str) -> list:
        return ["voice"]

    def map_openai_params(
        self,
        model: str,
        optional_params: dict,
        voice: str | dict | None = None,
        drop_params: bool = False,
        kwargs: dict = {},
    ) -> tuple[str | None, dict]:
        extra_body = (optional_params or {}).get("extra_body") or {}
        params = {
            **{
                k: v
                for k, v in (optional_params or {}).items()
                if v is not None and k not in UNSUPPORTED_OPENAI_PARAMS and k != "extra_body"
            },
            **{k: v for k, v in extra_body.items() if v is not None},
        }
        resolved_voice = voice if isinstance(voice, str) else None
        return resolved_voice, params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        final_api_key = api_key or litellm.api_key or get_secret_str("DASHSCOPE_API_KEY")
        if not final_api_key:
            raise ValueError("DASHSCOPE_API_KEY is not set")
        return {
            **headers,
            "Authorization": f"Bearer {final_api_key}",
            "Content-Type": "application/json",
        }

    def get_error_class(self, error_message: str, status_code: int, headers: dict | Headers) -> BaseLLMException:
        return DashScopeError(
            status_code=status_code,
            message=error_message,
            headers=headers if isinstance(headers, Headers) else Headers(headers),
        )

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        return api_base or get_secret_str("DASHSCOPE_API_BASE_TTS") or DEFAULT_API_BASE

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> TextToSpeechRequestData:
        tts_input = {
            "text": input,
            **({"voice": voice} if voice else {}),
            **{k: v for k, v in (optional_params or {}).items() if v is not None},
        }
        return TextToSpeechRequestData(
            dict_body={"model": model, "input": tts_input},
            headers={"Content-Type": "application/json"},
        )

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> "HttpxBinaryResponseContent":
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        if raw_response.status_code != 200:
            raise self.get_error_class(
                error_message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        try:
            response_json = raw_response.json()
        except ValueError as e:
            raise self.get_error_class(
                error_message=f"Failed to parse DashScope TTS response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        if "code" in response_json and "output" not in response_json:
            raise self.get_error_class(
                error_message=str(response_json.get("message", response_json)),
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        audio_url = response_json.get("output", {}).get("audio", {}).get("url")
        if not audio_url:
            raise self.get_error_class(
                error_message=f"No audio url in DashScope TTS response: {response_json}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        audio_response = httpx.get(audio_url, timeout=60.0)
        if audio_response.status_code != 200:
            raise self.get_error_class(
                error_message=f"Failed to download DashScope audio file: {audio_response.text}",
                status_code=audio_response.status_code,
                headers=audio_response.headers,
            )

        clean_headers = {
            k: v
            for k, v in dict(audio_response.headers).items()
            if k.lower() not in ("content-encoding", "transfer-encoding", "content-length")
        }
        clean_headers["content-length"] = str(len(audio_response.content))

        return HttpxBinaryResponseContent(
            httpx.Response(
                status_code=200,
                headers=clean_headers,
                content=audio_response.content,
                request=audio_response.request,
            )
        )
