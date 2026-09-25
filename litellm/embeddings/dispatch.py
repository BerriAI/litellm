import inspect
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from types import MappingProxyType
from typing import Final, TypeAlias, cast  # noqa: TID251  # native binding selects a sync result or an async awaitable

from litellm import main
from litellm.rust_bridge.catalog import Route, RouteContext
from litellm.rust_bridge.dispatch import PublicDispatch, call_hook
from litellm.rust_bridge.embeddings.entrypoints import (
    NATIVE_AEMBEDDING,
    NATIVE_EMBEDDING,
    LiteLLMEmbeddingRequest,
)
from litellm.rust_bridge.public_call import bind, optional_mapping, optional_str, signature
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
_EMBEDDING_SIGNATURE: Final = signature(_PYTHON_EMBEDDING)


def _public_request(
    legacy: inspect.Signature, args: tuple[object, ...], kwargs: Mapping[str, object]
) -> LiteLLMEmbeddingRequest | None:
    fields: Final = bind(legacy, args, kwargs)
    if fields is None:
        return None
    model: Final = fields.get("model")
    if not isinstance(model, str):
        return None
    extra: Final = optional_mapping(fields.get("kwargs")) or MappingProxyType({})
    return LiteLLMEmbeddingRequest(
        model=model,
        input=fields.get("input"),
        api_key=optional_str(fields.get("api_key")),
        api_base=optional_str(fields.get("api_base")),
        custom_llm_provider=optional_str(fields.get("custom_llm_provider")),
        kwargs=extra,
    )


def _context(request: LiteLLMEmbeddingRequest) -> RouteContext:
    return RouteContext(Route.EMBEDDINGS, provider=request.custom_llm_provider, model=request.model)


_DISPATCH: Final = PublicDispatch(
    route=Route.EMBEDDINGS,
    request=lambda args, kwargs: _public_request(_EMBEDDING_SIGNATURE, args, kwargs),
    context=_context,
    bypass=lambda request: request.kwargs.get("aembedding") is True,
)

_ADISPATCH: Final = PublicDispatch(
    route=Route.EMBEDDINGS,
    request=lambda args, kwargs: _public_request(_EMBEDDING_SIGNATURE, args, kwargs),
    context=_context,
)


def embedding(
    *args: object,
    **kwargs: object,  # kwargs-ok: preserve the public embedding call shape
) -> EmbeddingResponse | Coroutine[object, object, EmbeddingResponse]:
    return _DISPATCH.run(
        args,
        kwargs,
        python=_PYTHON_EMBEDDING,
        binding=NATIVE_EMBEDDING,
        native=call_hook,
    )


async def aembedding(*args: object, **kwargs: object) -> EmbeddingResponse:  # kwargs-ok: preserve the public call shape
    return await _ADISPATCH.arun(
        args,
        kwargs,
        python=_PYTHON_AEMBEDDING,
        binding=NATIVE_AEMBEDDING,
        native=call_hook,
    )


embedding.__doc__ = _PYTHON_EMBEDDING.__doc__
embedding.__wrapped__ = _PYTHON_EMBEDDING  # pyright: ignore[reportFunctionMemberAccess]  # preserve the legacy signature
aembedding.__doc__ = _PYTHON_AEMBEDDING.__doc__
aembedding.__wrapped__ = _PYTHON_AEMBEDDING  # pyright: ignore[reportFunctionMemberAccess]  # preserve the legacy signature
