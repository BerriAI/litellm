from typing import Final

import httpx
from openai import DefaultAsyncHttpxClient, DefaultHttpxClient
from openai import HttpxBinaryResponseContent as SDKBinaryResponse
from openai import Timeout as SDKTimeout
from openai._types import Response as SDKResponse
from typing_extensions import assert_type

import litellm
from litellm.exceptions import (
    APIConnectionError,
    APIResponseValidationError,
    BadGatewayError,
    InternalServerError,
    InvalidRequestError,
    RateLimitError,
    Timeout,
)
from litellm.litellm_core_utils.completion_timeout import CompletionTimeout
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.types.router import GenericLiteLLMParams, LiteLLMParamsTypedDict


def consume_binary_response(response: HttpxBinaryResponseContent) -> httpx.Response:
    assert_type(response.response, httpx.Response)
    assert_type(response.response.request, httpx.Request)
    return response.response


def binary_response_types(legacy_response: httpx.Response, sdk_response: SDKResponse) -> None:
    legacy: Final = HttpxBinaryResponseContent(legacy_response)
    native: Final = HttpxBinaryResponseContent(sdk_response)

    assert_type(legacy.response, httpx.Response)
    assert_type(native.response, SDKResponse)
    assert_type(legacy.read(), bytes)
    assert_type(native.read(), bytes)
    assert_type(consume_binary_response(legacy), httpx.Response)

    sdk_wrapper: Final[SDKBinaryResponse] = legacy
    assert_type(sdk_wrapper.read(), bytes)


def synthesized_response_types(
    error: RateLimitError | BadGatewayError | InternalServerError | APIResponseValidationError | InvalidRequestError,
) -> None:
    assert_type(error.response, httpx.Response)
    assert_type(error.request, httpx.Request)


def synthesized_request_types(error: APIConnectionError | Timeout) -> None:
    assert_type(error.request, httpx.Request)


def configured_clients(
    sync_client: httpx.Client | DefaultHttpxClient,
    async_client: httpx.AsyncClient | DefaultAsyncHttpxClient,
    timeout: httpx.Timeout | SDKTimeout,
) -> None:
    litellm.client_session = sync_client  # test-quality-ok: [TQ005] Type-check-only fixture; never executed
    litellm.aclient_session = async_client  # test-quality-ok: [TQ005] Type-check-only fixture; never executed
    assert_type(CompletionTimeout.normalize(timeout), httpx.Timeout)
    params: Final[LiteLLMParamsTypedDict] = {"timeout": 1.0}
    params["timeout"] = timeout
    GenericLiteLLMParams(timeout=timeout)
