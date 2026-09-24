"""
Map any exception litellm logs to one stable ``normalized_error`` code so dashboards can cluster
failures without parsing free-text messages that embed team names, token counts, model names, etc.
"""

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, Protocol, runtime_checkable

from litellm.exceptions import (
    APIConnectionError,
    AuthenticationError,
    BadGatewayError,
    BadRequestError,
    BlockedPiiEntityError,
    BudgetExceededError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    GuardrailRaisedException,
    InternalServerError,
    MidStreamFallbackError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    RateLimitType,
    ServiceUnavailableError,
    Timeout,
    UnprocessableEntityError,
    UnsupportedParamsError,
)

RATE_LIMIT_EXCEEDED: Final = "429_RATE_LIMIT_EXCEEDED"
BUDGET_EXCEEDED: Final = "429_BUDGET_EXCEEDED"
NO_HEALTHY_DEPLOYMENTS: Final = "429_NO_HEALTHY_DEPLOYMENTS"
AUTHENTICATION_FAILED: Final = "401_AUTHENTICATION_FAILED"
MODEL_ACCESS_DENIED: Final = "403_MODEL_ACCESS_DENIED"
PERMISSION_DENIED: Final = "403_PERMISSION_DENIED"
MISSING_REQUIRED_PARAMETER: Final = "400_MISSING_REQUIRED_PARAMETER"
INVALID_PARAMETER_VALUE: Final = "400_INVALID_PARAMETER_VALUE"
CONTEXT_WINDOW_EXCEEDED: Final = "400_CONTEXT_WINDOW_EXCEEDED"
CONTENT_POLICY_VIOLATION: Final = "400_CONTENT_POLICY_VIOLATION"
INVALID_REQUEST: Final = "400_INVALID_REQUEST"
RESOURCE_NOT_FOUND: Final = "404_RESOURCE_NOT_FOUND"
UPSTREAM_TIMEOUT: Final = "408_UPSTREAM_TIMEOUT"
PROVIDER_CONNECTION_ERROR: Final = "500_PROVIDER_CONNECTION_ERROR"
PROVIDER_OVERLOADED: Final = "503_PROVIDER_OVERLOADED"
PROVIDER_INTERNAL_ERROR: Final = "500_PROVIDER_INTERNAL_ERROR"
ROUTER_NO_FALLBACK: Final = "500_ROUTER_NO_FALLBACK"
ROUTER_FALLBACK_FAILURE: Final = "500_ROUTER_FALLBACK_FAILURE"
UPSTREAM_PASSTHROUGH: Final = "500_UPSTREAM_PASSTHROUGH"
UNSUPPORTED_OPERATION: Final = "500_UNSUPPORTED_OPERATION"
INTERNAL_STATE_ERROR: Final = "500_INTERNAL_STATE_ERROR"
UNCLASSIFIED: Final = "UNCLASSIFIED"


@runtime_checkable
class _HasProxyErrorType(Protocol):
    type: str


_MESSAGE_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (
        re.compile(r"budget has been exceeded|max budget|crossed budget", re.IGNORECASE),
        BUDGET_EXCEEDED,
    ),
    (re.compile(r"no healthy deployments?|no deployments available", re.IGNORECASE), NO_HEALTHY_DEPLOYMENTS),
    (re.compile(r"not allowed to access model due to tags configuration", re.IGNORECASE), MODEL_ACCESS_DENIED),
    (re.compile(r"upstream passthrough request failed", re.IGNORECASE), UPSTREAM_PASSTHROUGH),
    (re.compile(r"is not supported for provider|not implemented", re.IGNORECASE), UNSUPPORTED_OPERATION),
    (
        re.compile(r"context window|context length|(prompt|input) is too long|tokens? ?> ?\d+ ?maximum", re.IGNORECASE),
        CONTEXT_WINDOW_EXCEEDED,
    ),
    (re.compile(r"missing required parameter|field required", re.IGNORECASE), MISSING_REQUIRED_PARAMETER),
    (re.compile(r"overloaded|unable to process your request", re.IGNORECASE), PROVIDER_OVERLOADED),
    (
        re.compile(
            r"connection error|APIConnectionError|TransferEncodingError|payload is not completed|connection reset"
            r"|peer closed connection|incomplete chunked read",
            re.IGNORECASE,
        ),
        PROVIDER_CONNECTION_ERROR,
    ),
    (re.compile(r"timed? ?out", re.IGNORECASE), UPSTREAM_TIMEOUT),
)

_ROUTER_WRAPPER_PATTERNS: Final[tuple[tuple[re.Pattern[str], str], ...]] = (
    (re.compile(r"no fallback model group found", re.IGNORECASE), ROUTER_NO_FALLBACK),
    (re.compile(r"error doing the fallback|MidStreamFallbackError", re.IGNORECASE), ROUTER_FALLBACK_FAILURE),
)

_PROXY_ERROR_TYPE_MAP: Final[Mapping[str, str]] = MappingProxyType(
    {
        "budget_exceeded": BUDGET_EXCEEDED,
        "auth_error": AUTHENTICATION_FAILED,
        "expired_key": AUTHENTICATION_FAILED,
        "token_not_found_in_db": AUTHENTICATION_FAILED,
        "auth_provider_unavailable": AUTHENTICATION_FAILED,
        "key_model_access_denied": MODEL_ACCESS_DENIED,
        "team_model_access_denied": MODEL_ACCESS_DENIED,
        "user_model_access_denied": MODEL_ACCESS_DENIED,
        "org_model_access_denied": MODEL_ACCESS_DENIED,
        "project_model_access_denied": MODEL_ACCESS_DENIED,
        "agent_model_access_denied": MODEL_ACCESS_DENIED,
        "key_vector_store_access_denied": PERMISSION_DENIED,
        "team_vector_store_access_denied": PERMISSION_DENIED,
        "org_vector_store_access_denied": PERMISSION_DENIED,
        "tool_access_denied": PERMISSION_DENIED,
        "team_member_permission_error": PERMISSION_DENIED,
        "not_found_error": RESOURCE_NOT_FOUND,
    }
)

_STATUS_CODE_MAP: Final[Mapping[str, str]] = MappingProxyType(
    {
        "400": INVALID_REQUEST,
        "401": AUTHENTICATION_FAILED,
        "403": PERMISSION_DENIED,
        "404": RESOURCE_NOT_FOUND,
        "408": UPSTREAM_TIMEOUT,
        "422": INVALID_PARAMETER_VALUE,
        "429": RATE_LIMIT_EXCEEDED,
        "500": PROVIDER_INTERNAL_ERROR,
        "502": PROVIDER_INTERNAL_ERROR,
        "503": PROVIDER_OVERLOADED,
        "504": UPSTREAM_TIMEOUT,
    }
)

_INTERNAL_STATE_EXCEPTIONS: Final[tuple[type[BaseException], ...]] = (
    TypeError,
    KeyError,
    AttributeError,
    IndexError,
    RuntimeError,
    AssertionError,
    ZeroDivisionError,
)

_CLASS_CODE_TABLE: Final[tuple[tuple[tuple[type[BaseException], ...], str], ...]] = (
    ((AuthenticationError,), AUTHENTICATION_FAILED),
    ((PermissionDeniedError,), PERMISSION_DENIED),
    ((ContextWindowExceededError,), CONTEXT_WINDOW_EXCEEDED),
    ((ContentPolicyViolationError, GuardrailRaisedException, BlockedPiiEntityError), CONTENT_POLICY_VIOLATION),
    ((UnsupportedParamsError,), INVALID_PARAMETER_VALUE),
    ((NotFoundError,), RESOURCE_NOT_FOUND),
    ((Timeout,), UPSTREAM_TIMEOUT),
    ((MidStreamFallbackError,), ROUTER_FALLBACK_FAILURE),
    ((APIConnectionError,), PROVIDER_CONNECTION_ERROR),
    ((ServiceUnavailableError,), PROVIDER_OVERLOADED),
    ((InternalServerError, BadGatewayError), PROVIDER_INTERNAL_ERROR),
    ((BadRequestError, UnprocessableEntityError), INVALID_REQUEST),
    ((NotImplementedError,), UNSUPPORTED_OPERATION),
)


def _exceeded_before_budget(message: str) -> bool:
    """Linear-time equivalent of ``re.search(r"exceeded.*budget", message, re.IGNORECASE)``."""
    return any(
        (start := line.find("exceeded")) != -1 and line.find("budget", start + len("exceeded")) != -1
        for line in message.lower().split("\n")
    )


def _classify_by_message(message: str, patterns: tuple[tuple[re.Pattern[str], str], ...]) -> str | None:
    return next((code for pattern, code in patterns if pattern.search(message)), None)


def _classify_by_class(exc: Exception) -> str | None:
    if isinstance(exc, BudgetExceededError):
        return BUDGET_EXCEEDED
    if isinstance(exc, RateLimitError):
        return BUDGET_EXCEEDED if exc.rate_limit_type == RateLimitType.BUDGET.value else RATE_LIMIT_EXCEEDED
    for exc_types, code in _CLASS_CODE_TABLE:
        if isinstance(exc, exc_types):
            return code
    if isinstance(exc, _INTERNAL_STATE_EXCEPTIONS):
        return INTERNAL_STATE_ERROR
    return None


def normalize_error(exc: Exception | None, status_code: str, message: str) -> str | None:
    """
    Return a stable cluster key for ``exc``. ``status_code`` and ``message`` are the values
    ``get_error_information`` already extracted, so the same exception always yields the same code.
    """
    if exc is None:
        return None
    proxy_type: Final = exc.type if isinstance(exc, _HasProxyErrorType) else None
    by_proxy_type: Final = _PROXY_ERROR_TYPE_MAP.get(proxy_type) if isinstance(proxy_type, str) else None
    if by_proxy_type is not None:
        return by_proxy_type
    by_message: Final = (
        BUDGET_EXCEEDED if _exceeded_before_budget(message) else _classify_by_message(message, _MESSAGE_PATTERNS)
    )
    if by_message is not None:
        return by_message
    by_class: Final = _classify_by_class(exc)
    if by_class is not None:
        return by_class
    by_router_wrapper: Final = _classify_by_message(message, _ROUTER_WRAPPER_PATTERNS)
    if by_router_wrapper is not None:
        return by_router_wrapper
    return _STATUS_CODE_MAP.get(status_code, UNCLASSIFIED)
