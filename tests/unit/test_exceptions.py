from operator import attrgetter
from typing import Final

import httpx
import openai
import pytest

import litellm
from litellm.exceptions import (
    MidStreamFallbackError,
    OpenAIBadGatewayError,
    OpenAIConflictBadRequestError,
    OpenAIConflictError,
    OpenAIConnectionError,
    OpenAIGatewayTimeoutError,
    OpenAIServerErrorBadRequestError,
    OpenAIServiceUnavailableError,
    OpenAIUnprocessableEntityError,
    logged_class_name,
    reported_class_names,
)

_REQUEST: Final = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
_status_code: Final = attrgetter("status_code")
_UNSET: Final = object()


def _mro_names(cls: type) -> tuple[str, ...]:
    return tuple(klass.__name__ for klass in cls.__mro__)


def _reply(status: int) -> httpx.Response:
    return httpx.Response(
        status,
        json={"error": {"message": "upstream", "type": "upstream_type", "code": "c", "param": "p"}},
        headers={"x-request-id": "req-1"},
        request=_REQUEST,
    )


_BODY: Final = {"message": "upstream", "type": "upstream_type", "code": "c", "param": "p"}

_PARITY_CASES: Final = (
    (
        OpenAIConflictError(
            message="APIError: OpenAIException - conflict", llm_provider="openai", model="m", num_retries=2
        ),
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
        OpenAIConflictBadRequestError(
            message="OpenAIException - conflict", model="m", llm_provider="openai", response=_reply(409), body=_BODY
        ),
        litellm.BadRequestError(
            message="OpenAIException - conflict", model="m", llm_provider="openai", response=_reply(409), body=_BODY
        ),
        openai.ConflictError,
    ),
    (
        OpenAIServerErrorBadRequestError(
            message="OpenAIException - overloaded", model="m", llm_provider="openai", response=_reply(503), body=_BODY
        ),
        litellm.BadRequestError(
            message="OpenAIException - overloaded", model="m", llm_provider="openai", response=_reply(503), body=_BODY
        ),
        openai.InternalServerError,
    ),
    (
        OpenAIUnprocessableEntityError(
            message="OpenAIException - bad", model="m", llm_provider="openai", response=_reply(422), body=_BODY
        ),
        litellm.BadRequestError(
            message="OpenAIException - bad", model="m", llm_provider="openai", response=_reply(422), body=_BODY
        ),
        openai.UnprocessableEntityError,
    ),
    (
        OpenAIBadGatewayError(message="BadGatewayError: OpenAIException - x", llm_provider="openai", model="m"),
        litellm.BadGatewayError(message="BadGatewayError: OpenAIException - x", llm_provider="openai", model="m"),
        openai.InternalServerError,
    ),
    (
        OpenAIServiceUnavailableError(
            message="ServiceUnavailableError: OpenAIException - x", llm_provider="openai", model="m", max_retries=1
        ),
        litellm.ServiceUnavailableError(
            message="ServiceUnavailableError: OpenAIException - x", llm_provider="openai", model="m", max_retries=1
        ),
        openai.InternalServerError,
    ),
    (
        OpenAIGatewayTimeoutError(
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
        OpenAIConnectionError(
            message="InternalServerError: OpenAIException - Connection error.", model="m", llm_provider="openai"
        ),
        litellm.InternalServerError(
            message="InternalServerError: OpenAIException - Connection error.", model="m", llm_provider="openai"
        ),
        openai.APIConnectionError,
    ),
)

_PARITY_AND_REPLACED: Final = tuple((error, replaced) for error, replaced, _ in _PARITY_CASES)


@pytest.mark.parametrize(("error", "replaced", "sdk_class"), _PARITY_CASES)
def test_parity_error_is_the_sdk_class_and_the_class_it_replaced(
    error: openai.APIError, replaced: openai.APIError, sdk_class: type[openai.APIError]
) -> None:
    assert isinstance(error, sdk_class)
    assert isinstance(error, type(replaced))


@pytest.mark.parametrize(("error", "replaced"), _PARITY_AND_REPLACED)
def test_parity_error_reads_like_the_error_it_replaced(error: openai.APIError, replaced: openai.APIError) -> None:
    assert (str(error), repr(error), error.args) == (str(replaced), repr(replaced), replaced.args)
    assert (error.message, error.type, error.code, error.param, error.body) == (
        replaced.message,
        replaced.type,
        replaced.code,
        replaced.param,
        replaced.body,
    )
    assert _status_code(error) == _status_code(replaced)
    assert logged_class_name(error) == type(replaced).__name__


@pytest.mark.parametrize(("error", "replaced"), _PARITY_AND_REPLACED)
def test_parity_error_sets_every_attribute_the_replaced_error_sets(
    error: openai.APIError, replaced: openai.APIError
) -> None:
    replaced_fields: Final = {
        name: value for name, value in vars(replaced).items() if name not in {"response", "request"}
    }
    assert {name: vars(error).get(name, _UNSET) for name in replaced_fields} == replaced_fields


@pytest.mark.parametrize("error", tuple(error for error, _, _ in _PARITY_CASES))
def test_parity_error_carries_a_response_with_its_status(error: openai.APIStatusError) -> None:
    assert isinstance(error.response, httpx.Response)
    assert error.response.status_code == error.status_code
    assert error.request_id == error.response.headers.get("x-request-id")


@pytest.mark.parametrize(("error", "replaced"), _PARITY_AND_REPLACED)
def test_parity_error_reports_the_class_names_of_the_error_it_replaced(
    error: openai.APIError, replaced: openai.APIError
) -> None:
    assert reported_class_names(error) == _mro_names(type(replaced))


@pytest.mark.parametrize(
    "error",
    [
        litellm.BadGatewayError(message="bad gateway", llm_provider="anthropic", model="m"),
        litellm.ServiceUnavailableError(message="unavailable", llm_provider="bedrock", model="m"),
        MidStreamFallbackError(message="mid stream", llm_provider="openai", model="m"),
    ],
)
def test_5xx_errors_outside_the_openai_mapper_stay_out_of_the_sdk_internal_server_error(error: Exception) -> None:
    assert not isinstance(error, openai.InternalServerError)


@pytest.mark.parametrize(
    "error",
    [
        litellm.InternalServerError(message="boom", llm_provider="openai", model="m"),
        litellm.Timeout(message="slow", llm_provider="openai", model="m"),
        litellm.APIError(status_code=418, message="teapot", llm_provider="openai", model="m"),
        litellm.RateLimitError(message="slow down", llm_provider="openai", model="m"),
        litellm.APIConnectionError(message="refused", llm_provider="openai", model="m"),
        litellm.BadGatewayError(message="bad gateway", llm_provider="openai", model="m"),
        MidStreamFallbackError(message="mid stream", llm_provider="openai", model="m"),
        openai.InternalServerError("boom", response=httpx.Response(500, request=_REQUEST), body=None),
        openai.APIConnectionError(request=_REQUEST),
    ],
)
def test_reported_class_names_is_the_mro_for_errors_without_sdk_parity_bases(error: Exception) -> None:
    assert reported_class_names(error) == _mro_names(type(error))
    assert logged_class_name(error) == type(error).__name__


def test_logged_class_name_keeps_an_explicit_logged_error_class() -> None:
    refusal: Final = litellm.RequestOverDeploymentTpmError(message="too big", llm_provider="openai", model="m")

    assert logged_class_name(refusal) == "RateLimitError"
