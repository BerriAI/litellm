from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Final, TypeAlias, cast  # noqa: TID251  # validates dynamically loaded native callables

from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.public_call import NativeCall
from litellm.types.utils import ModelResponse

NativeCompletion: TypeAlias = Callable[[NativeCall], ModelResponse]

NativeAcompletion: TypeAlias = Callable[[NativeCall], Awaitable[ModelResponse]]


def _completion_binding(value: object) -> NativeCompletion | None:
    if not callable(value):
        return None
    return cast("NativeCompletion", value)  # cast-ok: callable validated at the native binding boundary


def _acompletion_binding(value: object) -> NativeAcompletion | None:
    if not callable(value):
        return None
    return cast("NativeAcompletion", value)  # cast-ok: callable validated at the native binding boundary


NATIVE_COMPLETION: Final = NativeBinding("completion", validate=_completion_binding)
NATIVE_ACOMPLETION: Final = NativeBinding("acompletion", validate=_acompletion_binding)
