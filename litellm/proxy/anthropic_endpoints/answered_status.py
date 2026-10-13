from typing import Final, Protocol, runtime_checkable

from litellm.anthropic_interface.exceptions import AnthropicExceptionMapping
from litellm.exceptions import InternalServerError, MidStreamFallbackError
from litellm.proxy._types import ProxyException


@runtime_checkable
class _CarriesStatusCode(Protocol):
    status_code: int


def _provider_status_code(exc: Exception | None) -> int | None:
    match exc:
        case InternalServerError(provider_status_code=int() as provider_status):
            return provider_status
        case MidStreamFallbackError(original_exception=original):
            return _provider_status_code(original)
        case _:
            return None


def with_answered_status(exc: Exception) -> Exception:
    """The same exception carrying the status the Anthropic routes answer with (a 503 and an upstream Anthropic
    overload answer as 529), so the proxy-level recorders the failure hook runs (the spend log row, the OTel server
    span, the failed-request metric, alerting) agree with the client. The deployment-level recorders (the deployment
    failure metric, cooldown, retry policy) ran in the litellm layer with the exception's own class and status
    before the exception gets here."""
    if isinstance(exc, ProxyException):
        if exc.code.isdecimal():
            answered_code: Final = str(AnthropicExceptionMapping.get_status_code(int(exc.code)))
            exc.code = answered_code  # rebind-ok: the recorders key on this object's class and attributes, so the status moves on it, not on a copy
        return exc
    if isinstance(exc, _CarriesStatusCode):
        answered_status: Final = AnthropicExceptionMapping.get_status_code(
            _provider_status_code(exc) or exc.status_code
        )
        exc.status_code = answered_status  # rebind-ok: litellm exception classes bind their status in __init__, so no new instance can carry 529 under the same class
    return exc
