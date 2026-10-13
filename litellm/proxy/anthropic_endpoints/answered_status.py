from typing import Final, Protocol, TypeVar, runtime_checkable

from litellm.anthropic_interface.exceptions import AnthropicExceptionMapping
from litellm.exceptions import InternalServerError, MidStreamFallbackError
from litellm.proxy._types import ProxyException

_Raised = TypeVar("_Raised", bound=Exception)


@runtime_checkable
class _CarriesStatusCode(Protocol):
    status_code: int


def _provider_status_code(exc: Exception) -> int | None:
    unwrapped: Final = exc.original_exception if isinstance(exc, MidStreamFallbackError) else exc
    match unwrapped:
        case InternalServerError(provider_status_code=int() as provider_status):
            return provider_status
        case _:
            return None


def _copy_carrying(exc: _Raised, attribute: str, value: int | str) -> _Raised:
    answered: Final = type(exc).__new__(type(exc), *exc.args)
    answered.__dict__.update({**exc.__dict__, attribute: value})
    answered.__cause__ = exc.__cause__
    answered.__context__ = exc.__context__
    answered.__suppress_context__ = exc.__suppress_context__
    return answered.with_traceback(exc.__traceback__)


def with_answered_status(exc: Exception) -> Exception:
    """A copy of the exception, same class, carrying the status the Anthropic routes answer with (a 503 and an
    upstream Anthropic overload answer as 529), so the proxy-level recorders the failure hook runs (the spend log
    row, the OTel server span, the failed-request metric, alerting) agree with the client. The raised object keeps
    its own status, because the litellm layer already handed it to the deployment-level recorders (the deployment
    failure metric, cooldown, retry policy, custom loggers), which may still read it."""
    if isinstance(exc, ProxyException):
        if not exc.code.isdecimal():
            return exc
        return _copy_carrying(exc, "code", str(AnthropicExceptionMapping.get_status_code(int(exc.code))))
    if isinstance(exc, _CarriesStatusCode):
        return _copy_carrying(
            exc, "status_code", AnthropicExceptionMapping.get_status_code(_provider_status_code(exc) or exc.status_code)
        )
    return exc
