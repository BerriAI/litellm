"""Anthropic error format type definitions."""

from typing_extensions import Literal, Required, TypedDict


# Known Anthropic error types
# Source: https://docs.anthropic.com/en/api/errors
AnthropicErrorType = Literal[
    "invalid_request_error",
    "authentication_error",
    "permission_error",
    "not_found_error",
    "request_too_large",
    "rate_limit_error",
    "api_error",
    "overloaded_error",
]


class AnthropicErrorDetail(TypedDict):
    """Inner error detail in Anthropic format."""

    type: AnthropicErrorType
    message: str


class AnthropicErrorResponse(TypedDict, total=False):
    """
    Anthropic-formatted error response.

    Format:
    {
        "type": "error",
        "error": {"type": "...", "message": "..."},
        "request_id": "req_..."  # optional
    }
    """

    type: Required[Literal["error"]]
    error: Required[AnthropicErrorDetail]
    request_id: str
