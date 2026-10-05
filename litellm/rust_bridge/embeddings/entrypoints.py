from __future__ import annotations

from collections.abc import Awaitable, Mapping
from dataclasses import dataclass
from typing import Final, Protocol, cast  # noqa: TID251  # validates dynamically loaded native callables

from litellm.rust_bridge.bindings import NativeBinding
from litellm.types.utils import EmbeddingResponse


@dataclass(frozen=True, slots=True)
class LiteLLMEmbeddingRequest:
    model: str
    input: object
    api_key: str | None
    api_base: str | None
    custom_llm_provider: str | None
    kwargs: Mapping[str, object]


class NativeEmbedding(Protocol):
    def __call__(
        self,
        request: LiteLLMEmbeddingRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
    ) -> EmbeddingResponse: ...


class NativeAembedding(Protocol):
    def __call__(
        self,
        request: LiteLLMEmbeddingRequest,
        args: tuple[object, ...],
        kwargs: Mapping[str, object],
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
