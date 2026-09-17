from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import httpx
import openai
from pydantic import TypeAdapter, ValidationError

import litellm
from litellm.llms.base_llm.ocr.transformation import PROVIDER_NATIVE_RESPONSE_KEY, OCRResponse
from litellm.rust_bridge import failures
from litellm.rust_bridge.ocr.entrypoints import LiteLLMOcrRequest

_RESPONSE_ADAPTER: Final = TypeAdapter(dict[str, object])
_UPSTREAM_ARGS: Final = TypeAdapter(tuple[int, str])
_UPSTREAM_HEADERS: Final = TypeAdapter(list[tuple[str, str]])


class UpstreamFailure(Exception):
    def __init__(self, response: httpx.Response, cause: Exception) -> None:
        super().__init__(str(cause))
        self.message: Final = str(cause)
        self.response: Final = response
        self.status_code: Final = response.status_code
        self.__cause__ = cause


def _upstream_failure(error: Exception) -> Exception:
    try:
        status, body = _UPSTREAM_ARGS.validate_python(error.args)
        headers: Final = _UPSTREAM_HEADERS.validate_python(getattr(error, "headers", None))
    except ValidationError:
        return error
    return UpstreamFailure(httpx.Response(status, content=body.encode(), headers=headers), error)


def response(value: Mapping[str, object]) -> OCRResponse:
    provider_native_response: Final = value.get(PROVIDER_NATIVE_RESPONSE_KEY)
    normalized: Final = OCRResponse.model_validate(
        MappingProxyType({key: item for key, item in value.items() if key != PROVIDER_NATIVE_RESPONSE_KEY})
    )
    if isinstance(provider_native_response, Mapping):
        normalized.set_provider_native_response(_RESPONSE_ADAPTER.validate_python(provider_native_response))
    return normalized


def arguments(request: LiteLLMOcrRequest) -> Mapping[str, object]:
    return request.kwargs


def map_failure(error: Exception, request: LiteLLMOcrRequest, request_provider: str) -> Exception:
    if getattr(error, "ocr_request_format_error", False):
        return litellm.UnsupportedParamsError(
            message=f"Invalid `req_format`: {request.kwargs.get('req_format')!r}. Expected 'native' or 'litellm'.",
            model=request.model.removeprefix(f"{request_provider}/"),
            llm_provider=request_provider,
        )
    original: Final = _upstream_failure(error)
    public_error: Final = failures.map_failure(original, request.model, request_provider, arguments(request))
    if isinstance(original, UpstreamFailure) and public_error.__context__ is original:
        public_error.__context__ = error
        if isinstance(public_error, openai.APIStatusError):
            public_error.response = original.response
            public_error.status_code = original.status_code
    return public_error
