from __future__ import annotations

from collections.abc import Awaitable
from typing import Final, Protocol, cast  # noqa: TID251  # validates dynamically loaded native callables

from litellm.rust_bridge import public_execution
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.public_call import NativeCall
from litellm.types.utils import EmbeddingResponse


class NativeEmbedding(Protocol):
    def __call__(
        self,
        call: NativeCall,
    ) -> EmbeddingResponse: ...


class NativeAembedding(Protocol):
    def __call__(
        self,
        call: NativeCall,
    ) -> Awaitable[EmbeddingResponse]: ...


def _embedding_binding(value: object) -> NativeEmbedding | None:
    if not callable(value):
        return None
    native: Final = public_execution.synchronous(value)
    return cast("NativeEmbedding", native)  # cast-ok: the public execution adapter returns the route's result


def _aembedding_binding(value: object) -> NativeAembedding | None:
    if not callable(value):
        return None
    native: Final = public_execution.asynchronous(value)
    return cast("NativeAembedding", native)  # cast-ok: the public execution adapter returns the route's result


NATIVE_EMBEDDING: Final = NativeBinding("embedding", validate=_embedding_binding)
NATIVE_AEMBEDDING: Final = NativeBinding("aembedding", validate=_aembedding_binding)
