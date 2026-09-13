"""
Common utilities and exceptions for the NVIDIA Riva STT provider
"""

from typing import Any, Final

from litellm.llms.base_llm.chat.transformation import BaseLLMException


class NvidiaRivaException(BaseLLMException):
    """
    Exception raised for NVIDIA Riva (gRPC) errors.

    ``status_code`` is an HTTP-equivalent code derived from the underlying
    gRPC ``StatusCode`` (when available) so that litellm's existing error
    classifiers (RateLimitError, AuthenticationError, etc.) keep working.
    """


# Mapping from grpc.StatusCode.name -> equivalent HTTP status code.
# Kept as a plain dict (rather than importing grpc enums) so this module is
# importable without grpc installed.
_GRPC_STATUS_CODE_TO_HTTP: Final[dict] = {
    "OK": 200,
    "CANCELLED": 499,
    "UNKNOWN": 500,
    "INVALID_ARGUMENT": 400,
    "DEADLINE_EXCEEDED": 504,
    "NOT_FOUND": 404,
    "ALREADY_EXISTS": 409,
    "PERMISSION_DENIED": 403,
    "RESOURCE_EXHAUSTED": 429,
    "FAILED_PRECONDITION": 400,
    "ABORTED": 409,
    "OUT_OF_RANGE": 400,
    "UNIMPLEMENTED": 501,
    "INTERNAL": 500,
    "UNAVAILABLE": 503,
    "DATA_LOSS": 500,
    "UNAUTHENTICATED": 401,
}


def _extract_grpc_status_name(error: Any) -> str | None:
    """
    Best-effort extraction of a gRPC StatusCode name from an arbitrary error.

    Works for ``grpc.RpcError`` instances (which expose ``.code()``) as well
    as plain exceptions whose string representation contains a status name.
    """
    code_fn: Final = getattr(error, "code", None)
    if callable(code_fn):
        try:
            code = code_fn()
        except Exception:
            code = None
        name: Final = getattr(code, "name", None)
        if isinstance(name, str):
            return name
    return None


def _extract_grpc_details(error: Any) -> str | None:
    """Best-effort extraction of a human-readable detail string from a gRPC error."""
    details_fn: Final = getattr(error, "details", None)
    if callable(details_fn):
        try:
            details = details_fn()
        except Exception:
            details = None
        if isinstance(details, str) and details:
            return details
    return None


def grpc_error_to_litellm_exception(error: Exception) -> NvidiaRivaException:
    """
    Convert a gRPC error (or any exception raised from the Riva client) into
    a ``NvidiaRivaException`` with an appropriate HTTP-equivalent status code.
    """
    status_name: Final = _extract_grpc_status_name(error)
    http_status: Final = _GRPC_STATUS_CODE_TO_HTTP.get(status_name or "", 500)

    detail: Final = _extract_grpc_details(error) or str(error)
    message = f"NVIDIA Riva gRPC error ({status_name}): {detail}" if status_name else f"NVIDIA Riva error: {detail}"
    return NvidiaRivaException(status_code=http_status, message=message)
