from typing import TypeVar

from litellm._auth.failure import NativeAuthRuntimeError

_T = TypeVar("_T")  # rebind-ok: TypeVar must remain a bare static type-variable binding


def require_native_fact(value: _T | None) -> _T:
    if value is None:
        raise NativeAuthRuntimeError("missing fact for native model authorization read")
    return value
