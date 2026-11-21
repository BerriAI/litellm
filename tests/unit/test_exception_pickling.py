"""Verify litellm exceptions survive pickle roundtrips (fixes #22812).

Without __reduce__, concurrent.futures.ProcessPoolExecutor crashes with
BrokenProcessPool instead of propagating the actual error.
Only unpickle bytes produced by these local fixtures, never external input.
"""

import pickle
import threading
from typing import Final

import httpx
import openai
import pytest

from litellm.exceptions import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    AuthenticationError,
    BadGatewayError,
    BadRequestError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    InternalServerError,
    InvalidRequestError,
    JSONSchemaValidationError,
    LiteLLMUnknownProvider,
    MidStreamFallbackError,
    MockException,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    RejectedRequestError,
    ServiceUnavailableError,
    Timeout,
    UnprocessableEntityError,
    UnsupportedParamsError,
)

_DUMMY_RESPONSE = httpx.Response(status_code=403, request=httpx.Request("POST", "https://x"))
_UNPROCESSABLE_RESPONSE = httpx.Response(status_code=422, request=httpx.Request("POST", "https://x"))


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (
            RateLimitError,
            dict(message="rate limited", llm_provider="openai", model="pickle-model"),
        ),
        (
            AuthenticationError,
            dict(message="bad key", llm_provider="openai", model="pickle-model"),
        ),
        (
            BadRequestError,
            dict(message="bad request", llm_provider="openai", model="pickle-model"),
        ),
        (
            UnprocessableEntityError,
            dict(
                message="unprocessable",
                llm_provider="openai",
                model="pickle-model",
                response=_UNPROCESSABLE_RESPONSE,
            ),
        ),
        (
            NotFoundError,
            dict(message="not found", llm_provider="openai", model="pickle-model"),
        ),
        (
            Timeout,
            dict(message="timeout", llm_provider="openai", model="pickle-model"),
        ),
        (
            InternalServerError,
            dict(message="server error", llm_provider="openai", model="pickle-model"),
        ),
        (
            ServiceUnavailableError,
            dict(message="unavailable", llm_provider="openai", model="pickle-model"),
        ),
        (
            BadGatewayError,
            dict(message="bad gateway", llm_provider="openai", model="pickle-model"),
        ),
        (
            APIConnectionError,
            dict(message="connection failed", llm_provider="openai", model="pickle-model"),
        ),
        (
            PermissionDeniedError,
            dict(
                message="denied",
                llm_provider="openai",
                model="pickle-model",
                response=_DUMMY_RESPONSE,
            ),
        ),
        (
            APIError,
            dict(
                status_code=500,
                message="api error",
                llm_provider="openai",
                model="pickle-model",
            ),
        ),
        (
            RejectedRequestError,
            dict(
                message="rejected",
                llm_provider="openai",
                model="pickle-model",
                request_data={"prompt": "test"},
            ),
        ),
        (
            ContextWindowExceededError,
            dict(message="too long", llm_provider="openai", model="pickle-model"),
        ),
        (
            ContentPolicyViolationError,
            dict(
                message="policy violation",
                llm_provider="openai",
                model="pickle-model",
            ),
        ),
        (
            MidStreamFallbackError,
            dict(
                message="midstream fallback",
                llm_provider="openai",
                model="pickle-model",
            ),
        ),
        (
            InvalidRequestError,
            dict(message="invalid request", llm_provider="openai", model="pickle-model"),
        ),
        (
            APIResponseValidationError,
            dict(message="bad response", llm_provider="openai", model="pickle-model"),
        ),
        (
            UnsupportedParamsError,
            dict(message="unsupported", llm_provider="openai", model="pickle-model"),
        ),
        (
            MockException,
            dict(
                status_code=418,
                message="mock",
                llm_provider="openai",
                model="pickle-model",
            ),
        ),
    ],
    ids=lambda x: x.__name__ if isinstance(x, type) else "",
)
def test_pickle_roundtrip(cls, kwargs):
    err = cls(**kwargs)
    restored = pickle.loads(pickle.dumps(err))

    assert type(restored) is cls
    assert restored.llm_provider == "openai"
    assert restored.model == "pickle-model"
    assert kwargs["message"] in str(restored)
    assert str(restored) == str(err)

    # Verify extra fields survive the roundtrip
    if "status_code" in kwargs:
        assert restored.status_code == kwargs["status_code"]
    if "request_data" in kwargs:
        assert restored.request_data == kwargs["request_data"]
    if cls is UnprocessableEntityError:
        assert restored.status_code == 422


def test_pickle_subclass_with_model_constructor() -> None:
    error: Final = LiteLLMUnknownProvider(model="pickle-model", custom_llm_provider="unsupported")
    restored: Final[object] = pickle.loads(pickle.dumps(error))

    assert isinstance(restored, LiteLLMUnknownProvider)
    assert restored.model == error.model
    assert restored.llm_provider == error.llm_provider
    assert str(restored) == str(error)


def test_pickle_subclass_with_schema_constructor() -> None:
    error: Final = JSONSchemaValidationError(
        model="pickle-model", llm_provider="openai", raw_response='{"bad":true}', schema='{"type":"object"}'
    )
    restored: Final[object] = pickle.loads(pickle.dumps(error))

    assert isinstance(restored, JSONSchemaValidationError)
    assert restored.raw_response == error.raw_response
    assert restored.schema == error.schema
    assert str(restored) == str(error)


def test_pickle_preserves_retry_and_rate_limit_metadata() -> None:
    error: Final = RateLimitError(
        message="rate limited",
        llm_provider="openai",
        model="pickle-model",
        litellm_debug_info="worker context",
        max_retries=3,
        num_retries=1,
        category="litellm_rate_limit",
        rate_limit_type="requests",
        headers={"retry-after": "3"},
        detail={"reason": "request limit"},
    )
    restored: Final[object] = pickle.loads(pickle.dumps(error))

    assert isinstance(restored, RateLimitError)
    assert restored.litellm_debug_info == error.litellm_debug_info
    assert restored.max_retries == error.max_retries
    assert restored.num_retries == error.num_retries
    assert restored.category == error.category
    assert restored.rate_limit_type == error.rate_limit_type
    assert restored.headers == error.headers
    assert restored.detail == error.detail
    assert str(restored) == str(error)


def test_pickle_preserves_sdk_cause() -> None:
    cause: Final = openai.APIConnectionError(
        message="network down", request=httpx.Request("POST", "https://example.test/")
    )
    error: Final = MidStreamFallbackError(
        message="stream interrupted", llm_provider="openai", model="pickle-model", original_exception=cause
    )
    restored: Final[object] = pickle.loads(pickle.dumps(error))

    assert isinstance(restored, MidStreamFallbackError)
    assert isinstance(restored.original_exception, openai.APIConnectionError)
    assert str(restored.original_exception) == str(cause)
    assert restored.original_exception.request.url == cause.request.url
    assert str(restored) == str(error)


def test_pickle_preserves_response_metadata_without_live_transport() -> None:
    response: Final = httpx.Response(
        400,
        request=httpx.Request("POST", "https://example.test/"),
        headers={"x-request-id": "pickle-request"},
        content=b"bad request",
        extensions={"network_stream": threading.Lock()},
    )
    error: Final = BadRequestError(
        message="bad request", model="pickle-model", llm_provider="openai", response=response
    )
    restored: Final[object] = pickle.loads(pickle.dumps(error))

    assert isinstance(restored, BadRequestError)
    assert restored.response.headers["x-request-id"] == error.response.headers["x-request-id"]
    assert restored.response.content == error.response.content
    assert restored.request.method == error.request.method
    assert restored.request.url == error.request.url
