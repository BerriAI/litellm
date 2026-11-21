"""
Support for OpenAI's `/v1/audio/speech` endpoint on deAPI. The answer is raw audio, so the real
per-request cost travels in the `X-DeAPI-Cost` response header.

Docs: https://docs.deapi.ai
"""

from typing import TYPE_CHECKING, Final

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.text_to_speech.transformation import BaseTextToSpeechConfig, TextToSpeechRequestData
from litellm.types.llms.openai import HttpxBinaryResponseContent

from ..common_utils import DeapiException, endpoint_url, header_cost, json_headers

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_SUPPORTED_PARAMS: Final = ("voice", "response_format", "speed", "instructions")


class DeapiTextToSpeechConfig(BaseTextToSpeechConfig):
    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: inherited contract
        return list(_SUPPORTED_PARAMS)

    def map_openai_params(
        self,
        model: str,
        optional_params: dict[str, object],  # mutable-ok: inherited contract
        voice: str | dict[str, object] | None = None,  # mutable-ok: inherited contract
        drop_params: bool = False,
        kwargs: dict[str, object] | None = None,  # mutable-ok: inherited contract
    ) -> tuple[str | None, dict[str, object]]:  # mutable-ok: inherited contract
        return (voice if isinstance(voice, str) else None), optional_params

    def validate_environment(
        self,
        headers: dict[str, object],  # mutable-ok: inherited contract
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, object]:  # mutable-ok: inherited contract
        return json_headers(headers, api_key, model)

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict[str, object],  # mutable-ok: inherited contract
    ) -> str:
        return endpoint_url(api_base, "audio/speech")

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: dict[str, object],  # mutable-ok: inherited contract
        litellm_params: dict[str, object],  # mutable-ok: inherited contract
        headers: dict[str, object],  # mutable-ok: inherited contract
    ) -> TextToSpeechRequestData:
        fields: Final = (("model", model), ("input", input), ("voice", voice), *optional_params.items())
        return TextToSpeechRequestData(dict_body={key: value for key, value in fields if value is not None})

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> HttpxBinaryResponseContent:
        response: Final = HttpxBinaryResponseContent(response=raw_response)
        response.set_response_cost(header_cost(raw_response.headers))
        return response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, object] | httpx.Headers,  # mutable-ok: inherited contract
    ) -> BaseLLMException:
        return DeapiException(message=error_message, status_code=status_code, headers=headers)
