from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeAlias

import httpx
from pydantic import ConfigDict, TypeAdapter

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.text_to_speech.transformation import BaseTextToSpeechConfig, TextToSpeechRequestData
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared HTTP factory lacks typed params
    get_httpx_client,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # shared HTTP factory is private
)
from litellm.llms.replicate.common_utils import (
    ReplicateError,
    async_replicate_download,
    async_wait_for_replicate_prediction,
    replicate_api_base,
    replicate_auth_headers,
    replicate_download,
    replicate_model_owner,
    replicate_output_urls,
    replicate_positive_int,
    replicate_prediction_failure,
    replicate_prediction_metrics,
    replicate_prediction_request,
    wait_for_replicate_prediction,
)
from litellm.types.utils import LlmProviders, Usage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.llms.openai import HttpxBinaryResponseContent

_SpeechParams: TypeAlias = dict[str, object]
_SpeechHeaders: TypeAlias = dict[str, str]
_NATIVE_INPUT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))
_VOICE_KEY_BY_OWNER: Final[Mapping[str, str]] = MappingProxyType(
    {"minimax": "voice_id", "inworld": "voice_id", "qwen": "speaker"}
)


def _voice_key(model: str) -> str:
    return _VOICE_KEY_BY_OWNER.get(replicate_model_owner(model), "voice")


def _get_replicate_httpx_client(ssl_verify: object) -> HTTPHandler:
    return get_httpx_client(params={"ssl_verify": ssl_verify})


def _get_replicate_async_httpx_client(ssl_verify: object) -> AsyncHTTPHandler:
    return get_async_httpx_client(llm_provider=LlmProviders.REPLICATE, params={"ssl_verify": ssl_verify})


def _ssl_verify(logging_obj: "LiteLLMLoggingObj") -> object:
    litellm_params: Final = getattr(logging_obj, "litellm_params", None)
    return litellm_params.get("ssl_verify") if isinstance(litellm_params, Mapping) else None


class ReplicateTextToSpeechConfig(BaseTextToSpeechConfig):
    def __init__(
        self,
        sync_client_factory: Callable[[object], HTTPHandler] = _get_replicate_httpx_client,
        async_client_factory: Callable[[object], AsyncHTTPHandler] = _get_replicate_async_httpx_client,
    ) -> None:
        super().__init__()
        self._sync_client_factory: Final = sync_client_factory
        self._async_client_factory: Final = async_client_factory

    def get_supported_openai_params(
        self, model: str
    ) -> list[str]:  # mutable-ok: base class contract returns a plain list
        return ["voice", "speed"]

    def map_openai_params(
        self,
        model: str,
        optional_params: Mapping[str, object],
        voice: object = None,
        drop_params: bool = False,
        kwargs: Mapping[str, object] | None = None,
    ) -> tuple[str | None, _SpeechParams]:
        speed: Final = optional_params.get("speed")
        extra_body: Final = (kwargs or {}).get("extra_body")
        return (
            voice if isinstance(voice, str) and voice else None,
            {
                **(_NATIVE_INPUT.validate_python(extra_body) if isinstance(extra_body, Mapping) else {}),
                **({"speed": speed} if isinstance(speed, (int, float)) and not isinstance(speed, bool) else {}),
            },
        )

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> _SpeechHeaders:
        return {**replicate_auth_headers(headers, api_key), "Prefer": "wait"}

    def get_complete_url(self, model: str, api_base: str | None, litellm_params: Mapping[str, object]) -> str:
        return replicate_prediction_request(replicate_api_base(api_base), model, MappingProxyType({})).url

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, str],
    ) -> TextToSpeechRequestData:
        prediction_input: Final = {
            **optional_params,
            **({_voice_key(model): voice} if voice else {}),
            "text": input,
        }
        return {"dict_body": dict(replicate_prediction_request("", model, prediction_input).body)}

    @staticmethod
    def _audio_url(prediction: Mapping[str, object], raw_response: httpx.Response) -> str:
        failure: Final = replicate_prediction_failure(prediction)
        if failure is not None:
            raise ReplicateError(status_code=raw_response.status_code, message=failure, headers=raw_response.headers)
        urls: Final = replicate_output_urls(prediction.get("output"))
        if not urls:
            raise ReplicateError(
                status_code=raw_response.status_code,
                message="Replicate prediction finished without audio output",
                headers=raw_response.headers,
            )
        return urls[0]

    @staticmethod
    def _usage(prediction: Mapping[str, object]) -> Usage | None:
        metrics: Final = replicate_prediction_metrics(prediction)
        prompt_tokens: Final = replicate_positive_int(
            metrics.get("token_input_count", metrics.get("input_token_count"))
        )
        completion_tokens: Final = replicate_positive_int(
            metrics.get("token_output_count", metrics.get("output_token_count"))
        )
        if prompt_tokens is None and completion_tokens is None:
            return None
        return Usage(
            prompt_tokens=prompt_tokens or 0,
            completion_tokens=completion_tokens or 0,
            total_tokens=(prompt_tokens or 0) + (completion_tokens or 0),
        )

    @staticmethod
    def _binary_response(
        audio_response: httpx.Response, raw_response: httpx.Response, usage: Usage | None
    ) -> "HttpxBinaryResponseContent":
        from litellm.types.llms.openai import HttpxBinaryResponseContent

        audio_response.raise_for_status()
        return HttpxBinaryResponseContent(
            httpx.Response(
                status_code=200,
                headers={
                    "content-type": audio_response.headers.get("content-type", "application/octet-stream"),
                    "content-length": str(len(audio_response.content)),
                },
                content=audio_response.content,
                request=raw_response.request,
            ),
            usage=usage,
        )

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> "HttpxBinaryResponseContent":
        client: Final = self._sync_client_factory(_ssl_verify(logging_obj))
        prediction: Final = wait_for_replicate_prediction(raw_response, client)
        audio_url: Final = self._audio_url(prediction, raw_response)
        return self._binary_response(replicate_download(client, audio_url), raw_response, self._usage(prediction))

    async def async_transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> "HttpxBinaryResponseContent":
        client: Final = self._async_client_factory(_ssl_verify(logging_obj))
        prediction: Final = await async_wait_for_replicate_prediction(raw_response, client)
        audio_url: Final = self._audio_url(prediction, raw_response)
        return self._binary_response(
            await async_replicate_download(client, audio_url), raw_response, self._usage(prediction)
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: _SpeechHeaders | httpx.Headers,
    ) -> BaseLLMException:
        return ReplicateError(status_code=status_code, message=error_message, headers=headers)
