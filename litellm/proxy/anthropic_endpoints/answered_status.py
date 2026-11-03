from typing import Final, TypeVar

from litellm.exceptions import InternalServerError, MidStreamFallbackError

_Raised = TypeVar("_Raised", bound=Exception)


def _provider_status_code(exc: Exception) -> int | None:
    unwrapped = exc  # rebind-ok: the recursion gate requires iterative wrapper traversal
    while isinstance(unwrapped, MidStreamFallbackError) and unwrapped.original_exception is not None:
        unwrapped = unwrapped.original_exception
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
    provider_status_code: Final = _provider_status_code(exc)
    return exc if provider_status_code is None else _copy_carrying(exc, "status_code", provider_status_code)
