from __future__ import annotations

from collections.abc import Awaitable
from typing import Final, Protocol, cast  # noqa: TID251  # validates dynamically loaded native callables

from litellm.rust_bridge import public_execution
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.public_call import NativeCall
from litellm.types.utils import ModelResponse


class NativeCompletion(Protocol):
    def __call__(
        self,
        call: NativeCall,
    ) -> ModelResponse: ...


class NativeAcompletion(Protocol):
    def __call__(
        self,
        call: NativeCall,
    ) -> Awaitable[ModelResponse]: ...


def _completion_binding(value: object) -> NativeCompletion | None:
    if not callable(value):
        return None
    native: Final = public_execution.synchronous(value)
    return cast("NativeCompletion", native)  # cast-ok: the public execution adapter returns the route's result


def _acompletion_binding(value: object) -> NativeAcompletion | None:
    if not callable(value):
        return None
    native: Final = public_execution.asynchronous(value)
    return cast("NativeAcompletion", native)  # cast-ok: the public execution adapter returns the route's result


NATIVE_COMPLETION: Final = NativeBinding("completion", validate=_completion_binding)
NATIVE_ACOMPLETION: Final = NativeBinding("acompletion", validate=_acompletion_binding)
