from __future__ import annotations

from collections.abc import Awaitable
from typing import Final, Protocol, cast  # noqa: TID251  # validates dynamically loaded native callables

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
    return cast("NativeEmbedding", value)  # cast-ok: callable validated at the native binding boundary


def _aembedding_binding(value: object) -> NativeAembedding | None:
    if not callable(value):
        return None
    return cast("NativeAembedding", value)  # cast-ok: callable validated at the native binding boundary


NATIVE_EMBEDDING: Final = NativeBinding("embedding", validate=_embedding_binding)
NATIVE_AEMBEDDING: Final = NativeBinding("aembedding", validate=_aembedding_binding)
