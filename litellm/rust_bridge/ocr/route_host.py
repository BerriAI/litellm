from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import httpx
import openai
from pydantic import TypeAdapter

import litellm
from litellm.llms.base_llm.ocr.transformation import PROVIDER_NATIVE_RESPONSE_KEY, OCRResponse
from litellm.rust_bridge import failures
from litellm.rust_bridge.ocr.entrypoints import LiteLLMOcrRequest

_RESPONSE_ADAPTER: Final = TypeAdapter(dict[str, object])


class UpstreamFailure(Exception):
    """A provider answer, shaped for litellm's public exception mapper."""

    def __init__(self, response: httpx.Response, cause: Exception) -> None:
        super().__init__(response.text)
        self.message: Final = response.text
        self.response: Final = response
        self.status_code: Final = response.status_code
        self.__cause__ = cause


class RequestFailure(Exception):
    """A native rejection of the request itself, shaped as the 400 the mapper expects."""

    def __init__(self, message: str, cause: Exception) -> None:
        super().__init__(message)
        self.message: Final = message
        self.status_code: Final = 400
        self.__cause__ = cause


class NativeFailure(Exception):
    """Any other native failure: no status, so the mapper reports a connection error."""

    def __init__(self, message: str, cause: Exception) -> None:
        super().__init__(message)
        self.message: Final = message
        self.__cause__ = cause


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


def _mapper_input(error: Exception, failure: failures.Declined | failures.Upstream | None) -> Exception:
    match failure:
        case failures.Upstream(status=0, body=body):
            return NativeFailure(body, error)
        case failures.Upstream(status=status, body=body, headers=headers):
            return UpstreamFailure(httpx.Response(status, content=body.encode(), headers=list(headers)), error)
        case failures.Declined(rejection="invalid_request", message=message):
            return RequestFailure(message, error)
        case failures.Declined(message=message):
            return NativeFailure(message, error)
        case None:
            return error


def map_failure(error: Exception, request: LiteLLMOcrRequest, request_provider: str) -> Exception:
    failure: Final = failures.native_failure(error)
    if isinstance(failure, failures.Declined) and failure.rejection == "request_format":
        return litellm.UnsupportedParamsError(
            message=f"Invalid `req_format`: {request.kwargs.get('req_format')!r}. Expected 'native' or 'litellm'.",
            model=request.model.removeprefix(f"{request_provider}/"),
            llm_provider=request_provider,
        )
    original: Final = _mapper_input(error, failure)
    public_error: Final = failures.map_failure(original, request.model, request_provider, arguments(request))
    if original is not error and public_error.__context__ is original:
        public_error.__context__ = error
        if isinstance(original, UpstreamFailure) and isinstance(public_error, openai.APIStatusError):
            public_error.response = original.response
            public_error.status_code = original.status_code
    return public_error
