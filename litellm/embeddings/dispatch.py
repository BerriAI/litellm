from collections.abc import Awaitable, Callable, Coroutine
from typing import Final, TypeAlias, cast  # noqa: TID251  # native binding selects a sync result or an async awaitable

from litellm import main
from litellm.rust_bridge.catalog import Route
from litellm.rust_bridge.dispatch import PublicDispatch, model_is_named
from litellm.rust_bridge.embeddings.entrypoints import (
    NATIVE_AEMBEDDING,
    NATIVE_EMBEDDING,
)
from litellm.rust_bridge.public_call import binder, native_call_hook
from litellm.types.utils import EmbeddingResponse

__all__ = ("aembedding", "embedding")

PythonEmbedding: TypeAlias = Callable[..., EmbeddingResponse | Coroutine[object, object, EmbeddingResponse]]
PythonAembedding: TypeAlias = Callable[..., Awaitable[EmbeddingResponse]]

_PYTHON_EMBEDDING: Final = cast(  # cast-ok: [LIT006] preserve the legacy public callable contract
    PythonEmbedding, main.embedding
)
_PYTHON_AEMBEDDING: Final = cast(  # cast-ok: [LIT006] preserve the legacy public callable contract
    PythonAembedding, main.aembedding
)


_DISPATCH: Final = PublicDispatch(
    Route.EMBEDDINGS, bind=binder(_PYTHON_EMBEDDING), internal_hop="aembedding", accepts=model_is_named
)
_ADISPATCH: Final = PublicDispatch(Route.EMBEDDINGS, bind=binder(_PYTHON_EMBEDDING), accepts=model_is_named)


def embedding(
    *args: object,
    **kwargs: object,  # kwargs-ok: preserve the public embedding call shape
) -> EmbeddingResponse | Coroutine[object, object, EmbeddingResponse]:
    return _DISPATCH.run(
        args,
        kwargs,
        python=_PYTHON_EMBEDDING,
        binding=NATIVE_EMBEDDING,
        native=native_call_hook,
    )


async def aembedding(*args: object, **kwargs: object) -> EmbeddingResponse:  # kwargs-ok: preserve the public call shape
    return await _ADISPATCH.arun(
        args,
        kwargs,
        python=_PYTHON_AEMBEDDING,
        binding=NATIVE_AEMBEDDING,
        native=native_call_hook,
    )


embedding.__doc__ = _PYTHON_EMBEDDING.__doc__
embedding.__wrapped__ = _PYTHON_EMBEDDING  # pyright: ignore[reportFunctionMemberAccess]  # preserve the legacy signature
aembedding.__doc__ = _PYTHON_AEMBEDDING.__doc__
aembedding.__wrapped__ = _PYTHON_AEMBEDDING  # pyright: ignore[reportFunctionMemberAccess]  # preserve the legacy signature
