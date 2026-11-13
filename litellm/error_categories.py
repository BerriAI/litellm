"""
Protocol-level error categorization for provider-agnostic retry/circuit-breaker logic.

Each provider adapter maps its native errors into exactly four canonical categories:
  auth       — authentication/authorization failure (401, 403)
  rate_limit — rate limit exceeded (429)
  server     — upstream server error (5xx)
  client     — invalid request or client-side error (4xx, excluding 401/403/429)

This module is the Python equivalent of zeshim's `ParsedError` type and the
`parseError` protocol function — adapted to LiteLLM's existing exception hierarchy.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ErrorCategory(str, Enum):
    """Canonical error categories, provider-agnostic.

    Upstream retry/circuit-breaker/scheduler logic should branch on these
    four values rather than inspecting provider-specific exception types.
    """

    AUTH = "auth"
    RATE_LIMIT = "rate_limit"
    SERVER = "server"
    CLIENT = "client"


@dataclass(frozen=True)
class ParsedError:
    """Normalized error produced by a protocol's error parser.

    Each protocol adapter (anthropic, openai, gemini, etc.) implements a
    parse_error(data: dict, status: int) -> ParsedError function that maps
    provider-specific error shapes into this canonical form.
    """

    category: ErrorCategory
    message: Optional[str] = None
    status_code: Optional[int] = None


# ── Protocol-level parse_error functions ──


def _extract_error_body(data: dict) -> dict:
    """Normalize the error object from various provider shapes."""
    err = data.get("error", data)
    return err if isinstance(err, dict) else {}


def default_parse_error(data: dict, status: int) -> ParsedError:
    """Default HTTP-status-based error parser.

    Used by OpenAI-compatible and Anthropic protocols where HTTP status
    codes alone are sufficient for categorization.
    """
    err = _extract_error_body(data)
    message = err.get("message")

    if status in (401, 403):
        return ParsedError(
            category=ErrorCategory.AUTH, message=message, status_code=status
        )
    if status == 429:
        return ParsedError(
            category=ErrorCategory.RATE_LIMIT, message=message, status_code=status
        )
    if status >= 500:
        return ParsedError(
            category=ErrorCategory.SERVER,
            message=message or "Server error",
            status_code=status,
        )
    return ParsedError(
        category=ErrorCategory.CLIENT,
        message=message or "Client error",
        status_code=status,
    )


def google_parse_error(data: dict, status: int) -> ParsedError:
    """Google Generative AI / Vertex AI error parser.

    Google returns error status strings in the response body (e.g.
    'UNAUTHENTICATED', 'PERMISSION_DENIED', 'RESOURCE_EXHAUSTED',
    'UNAVAILABLE', 'DEADLINE_EXCEEDED') that override HTTP status
    for categorization.
    """
    err = _extract_error_body(data)
    message = err.get("message")
    google_status = err.get("status", "").upper()

    if status in (401, 403) or google_status in (
        "UNAUTHENTICATED",
        "PERMISSION_DENIED",
    ):
        return ParsedError(
            category=ErrorCategory.AUTH, message=message, status_code=status
        )
    if status == 429 or google_status == "RESOURCE_EXHAUSTED":
        return ParsedError(
            category=ErrorCategory.RATE_LIMIT, message=message, status_code=status
        )
    if status >= 500 or google_status in (
        "UNAVAILABLE",
        "INTERNAL",
        "DEADLINE_EXCEEDED",
    ):
        return ParsedError(
            category=ErrorCategory.SERVER,
            message=message or "Server error",
            status_code=status,
        )
    return ParsedError(
        category=ErrorCategory.CLIENT,
        message=message or "Client error",
        status_code=status,
    )


# ── Integration with LiteLLM's existing ProviderError ──


def categorize_exception(exc: Exception) -> Optional[ErrorCategory]:
    """Extract canonical ErrorCategory from any LiteLLM exception.

    Returns None if the exception cannot be categorized (caller should
    treat as CLIENT or re-raise).
    """
    # If the exception already carries a category attribute, use it.
    category = getattr(exc, "error_category", None)
    if isinstance(category, ErrorCategory):
        return category

    # Try status-code-based inference
    status = getattr(exc, "status_code", None)
    if status is not None:
        category_from_status = _categorize_by_status_code(status)
        if category_from_status is not None:
            return category_from_status

    # Fall back to type-name heuristics
    return _categorize_by_exception_name(exc)


def _categorize_by_status_code(status: any) -> Optional[ErrorCategory]:
    """Categorize error by HTTP status code.

    Handles both integer and string status codes.
    """
    # Normalize to integer
    if not isinstance(status, int):
        try:
            status = int(status)
        except (ValueError, TypeError):
            return None

    # Auth errors
    if status in (401, 403):
        return ErrorCategory.AUTH

    # Rate limiting
    if status == 429:
        return ErrorCategory.RATE_LIMIT

    # Server errors (including 408 Request Timeout which should be retryable)
    if status == 408 or status >= 500:
        return ErrorCategory.SERVER

    # Client errors (4xx except auth and rate limit)
    if 400 <= status < 500:
        return ErrorCategory.CLIENT

    return None


def _categorize_by_exception_name(exc: Exception) -> Optional[ErrorCategory]:
    """Categorize error by exception class name patterns."""
    name = type(exc).__name__.lower()

    if "auth" in name:
        return ErrorCategory.AUTH

    if "rate" in name or "throttl" in name:
        return ErrorCategory.RATE_LIMIT

    if "server" in name or "service" in name or "timeout" in name:
        return ErrorCategory.SERVER

    return None
