"""Verify litellm exceptions survive pickle roundtrips (fixes #22812).

Without __reduce__, concurrent.futures.ProcessPoolExecutor crashes with
BrokenProcessPool instead of propagating the actual error.
"""

import pickle

import httpx
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

_DUMMY_RESPONSE = httpx.Response(
    status_code=403, request=httpx.Request("POST", "https://x")
)
_UNPROCESSABLE_RESPONSE = httpx.Response(
    status_code=422, request=httpx.Request("POST", "https://x")
)


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (
            RateLimitError,
            dict(message="rate limited", llm_provider="openai", model="gpt-4o"),
        ),
        (
            AuthenticationError,
            dict(message="bad key", llm_provider="openai", model="gpt-4o"),
        ),
        (
            BadRequestError,
            dict(message="bad request", llm_provider="openai", model="gpt-4o"),
        ),
        (
            UnprocessableEntityError,
            dict(
                message="unprocessable",
                llm_provider="openai",
                model="gpt-4o",
                response=_UNPROCESSABLE_RESPONSE,
            ),
        ),
        (
            NotFoundError,
            dict(message="not found", llm_provider="openai", model="gpt-4o"),
        ),
        (
            Timeout,
            dict(message="timeout", llm_provider="openai", model="gpt-4o"),
        ),
        (
            InternalServerError,
            dict(message="server error", llm_provider="openai", model="gpt-4o"),
        ),
        (
            ServiceUnavailableError,
            dict(message="unavailable", llm_provider="openai", model="gpt-4o"),
        ),
        (
            BadGatewayError,
            dict(message="bad gateway", llm_provider="openai", model="gpt-4o"),
        ),
        (
            APIConnectionError,
            dict(message="connection failed", llm_provider="openai", model="gpt-4o"),
        ),
        (
            PermissionDeniedError,
            dict(
                message="denied",
                llm_provider="openai",
                model="gpt-4o",
                response=_DUMMY_RESPONSE,
            ),
        ),
        (
            APIError,
            dict(
                status_code=500,
                message="api error",
                llm_provider="openai",
                model="gpt-4o",
            ),
        ),
        (
            RejectedRequestError,
            dict(
                message="rejected",
                llm_provider="openai",
                model="gpt-4o",
                request_data={"prompt": "test"},
            ),
        ),
        (
            ContextWindowExceededError,
            dict(message="too long", llm_provider="openai", model="gpt-4o"),
        ),
        (
            ContentPolicyViolationError,
            dict(
                message="policy violation",
                llm_provider="openai",
                model="gpt-4o",
            ),
        ),
        (
            MidStreamFallbackError,
            dict(
                message="midstream fallback",
                llm_provider="openai",
                model="gpt-4o",
            ),
        ),
        (
            InvalidRequestError,
            dict(message="invalid request", llm_provider="openai", model="gpt-4o"),
        ),
        (
            APIResponseValidationError,
            dict(message="bad response", llm_provider="openai", model="gpt-4o"),
        ),
        (
            UnsupportedParamsError,
            dict(message="unsupported", llm_provider="openai", model="gpt-4o"),
        ),
        (
            MockException,
            dict(
                status_code=418,
                message="mock",
                llm_provider="openai",
                model="gpt-4o",
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
    assert restored.model == "gpt-4o"
    assert kwargs["message"] in str(restored)
    assert restored._raw_message == kwargs["message"]

    # Verify extra fields survive the roundtrip
    if "status_code" in kwargs:
        assert restored.status_code == kwargs["status_code"]
    if "request_data" in kwargs:
        assert restored.request_data == kwargs["request_data"]
    if cls is UnprocessableEntityError:
        assert restored.status_code == 422
