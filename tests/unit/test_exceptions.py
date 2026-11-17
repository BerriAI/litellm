from operator import attrgetter
from typing import Final

import httpx
import openai
import pytest

import litellm
from litellm.exceptions import (
    ConflictError,
    GatewayTimeoutError,
    MidStreamFallbackError,
    UnprocessableEntityBadRequestError,
    UpstreamConnectionError,
    logged_class_name,
    router_class_names,
)

_REQUEST: Final = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


_status_code: Final = attrgetter("status_code")


def _mro_names(cls: type) -> tuple[str, ...]:
    return tuple(klass.__name__ for klass in cls.__mro__)


def _reply(status: int) -> httpx.Response:
    return httpx.Response(
        status,
        json={"error": {"message": "upstream", "type": "upstream_type", "code": "c", "param": "p"}},
        headers={"x-request-id": "req-1"},
        request=_REQUEST,
    )


_PAIRS: Final = (
    (
        ConflictError(message="APIError: OpenAIException - conflict", llm_provider="openai", model="m", num_retries=2),
        litellm.APIError(
            status_code=409,
            message="APIError: OpenAIException - conflict",
            llm_provider="openai",
            model="m",
            num_retries=2,
        ),
        openai.ConflictError,
    ),
    (
        UnprocessableEntityBadRequestError(
            message="OpenAIException - bad", model="m", llm_provider="openai", response=_reply(422), body={"code": "c"}
        ),
        litellm.BadRequestError(
            message="OpenAIException - bad", model="m", llm_provider="openai", response=_reply(422), body={"code": "c"}
        ),
        openai.UnprocessableEntityError,
    ),
    (
        GatewayTimeoutError(
            message="Timeout Error: OpenAIException - slow",
            model="m",
            llm_provider="openai",
            exception_status_code=504,
            max_retries=3,
        ),
        litellm.Timeout(
            message="Timeout Error: OpenAIException - slow",
            model="m",
            llm_provider="openai",
            exception_status_code=504,
            max_retries=3,
        ),
        openai.InternalServerError,
    ),
    (
        UpstreamConnectionError(
            message="InternalServerError: OpenAIException - Connection error.", model="m", llm_provider="openai"
        ),
        litellm.InternalServerError(
            message="InternalServerError: OpenAIException - Connection error.", model="m", llm_provider="openai"
        ),
        openai.APIConnectionError,
    ),
)


@pytest.mark.parametrize(("error", "before", "sdk_class"), _PAIRS)
def test_sdk_parity_error_reads_like_the_error_it_replaced(
    error: openai.APIError, before: openai.APIError, sdk_class: type[openai.APIError]
) -> None:
    assert isinstance(error, sdk_class)
    assert isinstance(error, type(before))
    assert (str(error), repr(error), error.args) == (str(before), repr(before), before.args)
    assert (error.message, error.type, error.code, error.param, error.body) == (
        before.message,
        before.type,
        before.code,
        before.param,
        before.body,
    )
    assert _status_code(error) == _status_code(before)
    assert logged_class_name(error) == type(before).__name__


@pytest.mark.parametrize(("error", "before", "sdk_class"), _PAIRS)
def test_sdk_parity_error_carries_a_response_with_its_status(
    error: openai.APIStatusError, before: openai.APIError, sdk_class: type[openai.APIError]
) -> None:
    assert isinstance(error.response, httpx.Response)
    assert error.response.status_code == error.status_code
    assert error.request_id == error.response.headers.get("x-request-id")


@pytest.mark.parametrize(("error", "before", "sdk_class"), _PAIRS)
def test_router_sees_the_class_names_of_the_error_it_replaced(
    error: openai.APIError, before: openai.APIError, sdk_class: type[openai.APIError]
) -> None:
    assert router_class_names(error) == _mro_names(type(before))


@pytest.mark.parametrize(
    ("error", "names_before"),
    [
        (
            litellm.BadGatewayError(message="bad gateway", llm_provider="openai", model="m"),
            ("BadGatewayError", *_mro_names(openai.APIStatusError)),
        ),
        (
            litellm.ServiceUnavailableError(message="unavailable", llm_provider="openai", model="m"),
            ("ServiceUnavailableError", *_mro_names(openai.APIStatusError)),
        ),
        (
            MidStreamFallbackError(message="mid stream", llm_provider="openai", model="m"),
            ("MidStreamFallbackError", "ServiceUnavailableError", *_mro_names(openai.APIStatusError)),
        ),
    ],
)
def test_5xx_errors_are_the_sdk_internal_server_error_but_the_router_sees_the_old_names(
    error: Exception, names_before: tuple[str, ...]
) -> None:
    assert isinstance(error, openai.InternalServerError)
    assert not isinstance(error, litellm.InternalServerError)
    assert router_class_names(error) == names_before


@pytest.mark.parametrize(
    "error",
    [
        litellm.InternalServerError(message="boom", llm_provider="openai", model="m"),
        litellm.Timeout(message="slow", llm_provider="openai", model="m"),
        litellm.APIError(status_code=418, message="teapot", llm_provider="openai", model="m"),
        litellm.RateLimitError(message="slow down", llm_provider="openai", model="m"),
        litellm.APIConnectionError(message="refused", llm_provider="openai", model="m"),
        openai.InternalServerError("boom", response=httpx.Response(500, request=_REQUEST), body=None),
        openai.APIConnectionError(request=_REQUEST),
    ],
)
def test_router_class_names_is_the_mro_for_errors_without_sdk_parity_bases(error: Exception) -> None:
    assert router_class_names(error) == _mro_names(type(error))
