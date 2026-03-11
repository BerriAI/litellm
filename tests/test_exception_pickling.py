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
    AuthenticationError,
    BadGatewayError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    RejectedRequestError,
    ServiceUnavailableError,
)

_DUMMY_RESPONSE = httpx.Response(
    status_code=403, request=httpx.Request("POST", "https://x")
)


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (RateLimitError, dict(message="rate limited", llm_provider="openai", model="gpt-4o")),
        (AuthenticationError, dict(message="bad key", llm_provider="openai", model="gpt-4o")),
        (BadRequestError, dict(message="bad request", llm_provider="openai", model="gpt-4o")),
        (NotFoundError, dict(message="not found", llm_provider="openai", model="gpt-4o")),
        (InternalServerError, dict(message="server error", llm_provider="openai", model="gpt-4o")),
        (ServiceUnavailableError, dict(message="unavailable", llm_provider="openai", model="gpt-4o")),
        (BadGatewayError, dict(message="bad gateway", llm_provider="openai", model="gpt-4o")),
        (APIConnectionError, dict(message="connection failed", llm_provider="openai", model="gpt-4o")),
        (PermissionDeniedError, dict(message="denied", llm_provider="openai", model="gpt-4o", response=_DUMMY_RESPONSE)),
        (APIError, dict(status_code=500, message="api error", llm_provider="openai", model="gpt-4o")),
        (RejectedRequestError, dict(message="rejected", llm_provider="openai", model="gpt-4o", request_data={"prompt": "test"})),
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

    # Verify extra fields survive the roundtrip
    if "status_code" in kwargs:
        assert restored.status_code == kwargs["status_code"]
    if "request_data" in kwargs:
        assert restored.request_data == kwargs["request_data"]
