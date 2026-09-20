from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlparse

import httpx

import litellm
from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)
from litellm.secret_managers.main import get_secret_str

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import HttpxBinaryResponseContent
else:
    LiteLLMLoggingObj: Final = Any
    HttpxBinaryResponseContent: Final = Any


class OpenrouterTextToSpeechConfig(BaseTextToSpeechConfig):
    DEFAULT_BASE_URL: Final = "https://openrouter.ai/api/v1"

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: override abstract method signature
        return ["voice", "response_format", "speed", "instructions"]  # mutable-ok: list required by base signature

    def map_openai_params(
        self,
        model: str,
        optional_params: dict,  # mutable-ok: override abstract method signature
        voice: str | dict | None = None,  # mutable-ok: override abstract method signature
        drop_params: bool = False,
        kwargs: dict | None = None,  # mutable-ok: override abstract method signature
    ) -> tuple[str | None, dict]:  # mutable-ok: override abstract method signature
        mapped_voice: Final[str | None] = voice if isinstance(voice, str) else None
        return mapped_voice, optional_params

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: override abstract method signature
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: override abstract method signature
        return headers

    def is_trusted_api_base(self, api_base: str | None) -> bool:
        if not api_base:
            return True
        parsed: Final = urlparse(api_base)
        scheme: Final = parsed.scheme.lower()
        if scheme != "https":
            configured_base: Final = get_secret_str("OPENROUTER_API_BASE") or litellm.api_base
            return bool(
                configured_base is not None
                and api_base.rstrip("/") == configured_base.rstrip("/")
                and parsed.hostname in ("localhost", "127.0.0.1")
            )
        configured_base: Final = get_secret_str("OPENROUTER_API_BASE") or litellm.api_base
        if configured_base is not None and api_base.rstrip("/") == configured_base.rstrip("/"):
            return True
        hostname: Final = parsed.hostname or ""
        return hostname == "openrouter.ai" or hostname.endswith(".openrouter.ai")

    def resolve_api_base_and_key(
        self,
        api_base: str | None = None,
        api_key: str | None = None,
        dynamic_api_key: str | None = None,
    ) -> tuple[str, str | None]:
        resolved_base: Final = (
            api_base or litellm.api_base or get_secret_str("OPENROUTER_API_BASE") or self.DEFAULT_BASE_URL
        )
        if api_key:
            return resolved_base, api_key
        if dynamic_api_key:
            return resolved_base, dynamic_api_key
        if self.is_trusted_api_base(api_base):
            server_key: Final = (
                litellm.api_key
                or litellm.openrouter_key
                or get_secret_str("OPENROUTER_API_KEY")
                or get_secret_str("OR_API_KEY")
            )
            return resolved_base, server_key
        return resolved_base, None

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,  # mutable-ok: override abstract method signature
    ) -> str:
        base: Final = api_base or litellm.api_base or get_secret_str("OPENROUTER_API_BASE") or self.DEFAULT_BASE_URL
        return f"{base.rstrip('/')}/audio/speech"

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: dict,  # mutable-ok: override abstract method signature
        litellm_params: dict,  # mutable-ok: override abstract method signature
        headers: dict,  # mutable-ok: override abstract method signature
    ) -> TextToSpeechRequestData:
        payload: Final[dict[str, object]] = {  # mutable-ok: request body dict
            "model": model,
            "input": input,
            "voice": voice,
            **optional_params,
        }
        return TextToSpeechRequestData(dict_body=payload, headers=headers)

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> "HttpxBinaryResponseContent":
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        return HttpxBinaryResponseContent(response=raw_response)
