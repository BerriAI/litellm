import base64
import math
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeAlias

import httpx
from pydantic import ConfigDict, TypeAdapter

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_generation.transformation import BaseImageGenerationConfig
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
    replicate_output_urls,
    replicate_positive_int,
    replicate_prediction_failure,
    replicate_prediction_input,
    replicate_prediction_request,
    wait_for_replicate_prediction,
)
from litellm.types.llms.openai import AllMessageValues, OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageObject, ImageResponse, LlmProviders

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer

_ImageParams: TypeAlias = dict[str, object]
_ImageHeaders: TypeAlias = dict[str, str]


def _aspect_ratio(size: object) -> Mapping[str, str]:
    if not isinstance(size, str) or size.lower().count("x") != 1:
        return MappingProxyType({})
    width_text, height_text = size.lower().split("x")
    if not (width_text.isdigit() and height_text.isdigit()) or int(width_text) <= 0 or int(height_text) <= 0:
        return MappingProxyType({})
    divisor: Final = math.gcd(int(width_text), int(height_text))
    return MappingProxyType({"aspect_ratio": f"{int(width_text) // divisor}:{int(height_text) // divisor}"})


_NATIVE_INPUT: Final = TypeAdapter(Mapping[str, object], config=ConfigDict(hide_input_in_errors=True))


def _get_replicate_httpx_client(ssl_verify: object) -> HTTPHandler:
    return get_httpx_client(params={"ssl_verify": ssl_verify})


def _get_replicate_async_httpx_client(ssl_verify: object) -> AsyncHTTPHandler:
    return get_async_httpx_client(llm_provider=LlmProviders.REPLICATE, params={"ssl_verify": ssl_verify})


class ReplicateImageGenerationConfig(BaseImageGenerationConfig):
    def __init__(
        self,
        sync_client_factory: Callable[[object], HTTPHandler] = _get_replicate_httpx_client,
        async_client_factory: Callable[[object], AsyncHTTPHandler] = _get_replicate_async_httpx_client,
    ) -> None:
        self._sync_client_factory: Final = sync_client_factory
        self._async_client_factory: Final = async_client_factory

    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIImageGenerationOptionalParams]:  # mutable-ok: base class contract returns a plain list
        return ["n", "response_format", "size"]

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: _ImageParams,
        model: str,
        drop_params: bool,
    ) -> _ImageParams:
        supported: Final = frozenset(self.get_supported_openai_params(model))
        unsupported: Final = tuple(key for key in non_default_params if key not in supported)
        if unsupported and not drop_params:
            raise ValueError(
                f"Parameters {', '.join(unsupported)} are not supported for model {model}. "
                f"Supported parameters are {sorted(supported)}. Set drop_params=True to drop unsupported parameters."
            )
        count: Final = non_default_params.get("n")
        response_format: Final = non_default_params.get("response_format")
        return {
            **optional_params,
            **({"num_outputs": count} if isinstance(count, int) and not isinstance(count, bool) else {}),
            **_aspect_ratio(non_default_params.get("size")),
            **({"response_format": response_format} if isinstance(response_format, str) else {}),
        }

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: _ImageParams,
        litellm_params: _ImageParams,
        stream: bool | None = None,
    ) -> str:
        return replicate_prediction_request(replicate_api_base(api_base), model, MappingProxyType({})).url

    def validate_environment(
        self,
        headers: _ImageHeaders,
        model: str,
        messages: list[AllMessageValues],
        optional_params: _ImageParams,
        litellm_params: _ImageParams,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> _ImageHeaders:
        return {**replicate_auth_headers(headers, api_key), "Prefer": "wait"}

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: _ImageParams,
        litellm_params: _ImageParams,
        headers: _ImageHeaders,
    ) -> _ImageParams:
        extra_body: Final = optional_params.get("extra_body")
        prediction_input: Final = {
            **{key: value for key, value in optional_params.items() if key not in ("response_format", "extra_body")},
            **(_NATIVE_INPUT.validate_python(extra_body) if isinstance(extra_body, Mapping) else {}),
            "prompt": prompt,
        }
        return dict(replicate_prediction_request("", model, prediction_input).body)

    @staticmethod
    def _image_urls(prediction: Mapping[str, object], raw_response: httpx.Response) -> tuple[str, ...]:
        failure: Final = replicate_prediction_failure(prediction)
        if failure is not None:
            raise ReplicateError(status_code=raw_response.status_code, message=failure, headers=raw_response.headers)
        urls: Final = replicate_output_urls(prediction.get("output"))
        if not urls:
            raise ReplicateError(
                status_code=raw_response.status_code,
                message="Replicate prediction finished without image output",
                headers=raw_response.headers,
            )
        return urls

    @staticmethod
    def _native_size(prediction: Mapping[str, object]) -> Mapping[str, str]:
        prediction_input: Final = replicate_prediction_input(prediction)
        width: Final = replicate_positive_int(prediction_input.get("width"))
        height: Final = replicate_positive_int(prediction_input.get("height"))
        return {"size": f"{width}x{height}"} if width is not None and height is not None else {}

    @staticmethod
    def _image_objects(urls: tuple[str, ...], contents: tuple[bytes, ...] | None) -> tuple[ImageObject, ...]:
        if contents is None:
            return tuple(ImageObject(url=url, b64_json=None) for url in urls)
        return tuple(ImageObject(url=None, b64_json=base64.b64encode(content).decode()) for content in contents)

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: _ImageParams,
        optional_params: _ImageParams,
        litellm_params: _ImageParams,
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        client: Final = self._sync_client_factory(litellm_params.get("ssl_verify"))
        prediction: Final = wait_for_replicate_prediction(raw_response, client)
        urls: Final = self._image_urls(prediction, raw_response)
        contents: Final = (
            tuple(self._download(client, url) for url in urls)
            if optional_params.get("response_format") == "b64_json"
            else None
        )
        return model_response.model_copy(
            update={"data": list(self._image_objects(urls, contents)), **self._native_size(prediction)}
        )

    async def async_transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: _ImageParams,
        optional_params: _ImageParams,
        litellm_params: _ImageParams,
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        client: Final = self._async_client_factory(litellm_params.get("ssl_verify"))
        prediction: Final = await async_wait_for_replicate_prediction(raw_response, client)
        urls: Final = self._image_urls(prediction, raw_response)
        contents: Final = (
            tuple([await self._async_download(client, url) for url in urls])
            if optional_params.get("response_format") == "b64_json"
            else None
        )
        return model_response.model_copy(
            update={"data": list(self._image_objects(urls, contents)), **self._native_size(prediction)}
        )

    @staticmethod
    def _download(client: HTTPHandler, url: str) -> bytes:
        return replicate_download(client, url).content

    @staticmethod
    async def _async_download(client: AsyncHTTPHandler, url: str) -> bytes:
        return (await async_replicate_download(client, url)).content

    def get_error_class(
        self, error_message: str, status_code: int, headers: _ImageHeaders | httpx.Headers
    ) -> BaseLLMException:
        return ReplicateError(status_code=status_code, message=error_message, headers=headers)
