import httpx
from typing_extensions import assert_type

from litellm.exceptions import (
    APIError,
    AuthenticationError,
    BadRequestError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    ImageFetchError,
    MidStreamFallbackError,
    MockException,
    NotFoundError,
    PermissionDeniedError,
    ServiceUnavailableError,
    UnprocessableEntityError,
    UnsupportedParamsError,
)


def legacy_response_types(
    error: AuthenticationError
    | BadRequestError
    | ContentPolicyViolationError
    | ContextWindowExceededError
    | ImageFetchError
    | MidStreamFallbackError
    | NotFoundError
    | PermissionDeniedError
    | ServiceUnavailableError
    | UnprocessableEntityError
    | UnsupportedParamsError,
) -> None:
    assert_type(error.response, httpx.Response)
    assert_type(error.request, httpx.Request)


def legacy_request_types(error: APIError | MockException) -> None:
    assert_type(error.request, httpx.Request)
