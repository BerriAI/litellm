from collections.abc import Callable
from functools import wraps
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")  # rebind-ok: ParamSpec declarations cannot be annotated Final
_R = TypeVar("_R")  # rebind-ok: TypeVar declarations cannot be annotated Final


class NativeAuthFailure(Exception):
    pass


class NativeAuthTypeError(NativeAuthFailure, TypeError):
    pass


class NativeAuthValueError(NativeAuthFailure, ValueError):
    pass


class NativeAuthRuntimeError(NativeAuthFailure, RuntimeError):
    pass


def native_boundary(function: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(function)
    def call(*args: _P.args, **kwargs: _P.kwargs) -> _R:  # kwargs-ok: ParamSpec preserves the wrapped signature
        try:
            return function(*args, **kwargs)
        except NativeAuthFailure:
            raise
        except TypeError as error:
            raise NativeAuthTypeError(str(error)) from error
        except ValueError as error:
            raise NativeAuthValueError(str(error)) from error
        except Exception as error:  # noqa: BLE001  # native faults must bypass legacy IO fail-open catches
            raise NativeAuthRuntimeError(str(error)) from error

    return call
