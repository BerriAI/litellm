from collections.abc import Awaitable
from typing import (
    Final,
    Protocol,
    cast,
)  # noqa: TID251  # validates dynamically loaded native callables

from litellm.rust_bridge import execution
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.public_call import NativeCall
from litellm.types.tokenize import TokenizeResponse


class NativeTokenize(Protocol):
    def __call__(self, request: NativeCall) -> TokenizeResponse: ...


class NativeAtokenize(Protocol):
    def __call__(self, request: NativeCall) -> Awaitable[TokenizeResponse]: ...


def _tokenize_binding(value: object) -> NativeTokenize | None:
    if not callable(value):
        return None
    return cast("NativeTokenize", execution.synchronous(value))  # cast-ok: execution returns the route's result


def _atokenize_binding(value: object) -> NativeAtokenize | None:
    if not callable(value):
        return None
    return cast("NativeAtokenize", execution.asynchronous(value))  # cast-ok: execution returns the route's result


NATIVE_TOKENIZE: Final = NativeBinding("tokenize", validate=_tokenize_binding)
NATIVE_ATOKENIZE: Final = NativeBinding("atokenize", validate=_atokenize_binding)
