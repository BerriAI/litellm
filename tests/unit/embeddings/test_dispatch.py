from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from typing import Final, cast  # noqa: TID251  # narrows legacy callable signatures

import pytest
from pydantic import TypeAdapter

import litellm
from litellm import main as python_embeddings
from litellm.embeddings.dispatch import (
    _ADISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
    _DISPATCH,  # pyright: ignore[reportPrivateUsage]  # tests configured dispatch
)
from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import Decision, Route, RouteContext, Rust
from litellm.rust_bridge.dispatch import PublicDispatch
from litellm.rust_bridge.embeddings.entrypoints import NATIVE_AEMBEDDING, NATIVE_EMBEDDING
from litellm.rust_bridge.public_call import NativeCall
from litellm.types.utils import EmbeddingResponse


def required_everywhere(_context: RouteContext) -> Decision:
    return Rust(required=True)


def test_public_signature_is_the_legacy_signature() -> None:
    public_embedding: Final = cast(Callable[..., object], litellm.embedding)
    legacy_embedding: Final = cast(Callable[..., object], python_embeddings.embedding)
    public_aembedding: Final = cast(Callable[..., object], litellm.aembedding)
    legacy_aembedding: Final = cast(Callable[..., object], python_embeddings.aembedding)
    assert inspect.signature(public_embedding) == inspect.signature(legacy_embedding)
    assert inspect.signature(public_aembedding) == inspect.signature(legacy_aembedding)


@pytest.mark.parametrize("dispatch", (_DISPATCH, _ADISPATCH), ids=("embedding", "aembedding"))
def test_request_binds_positional_and_keyword_arguments_onto_the_legacy_signature(dispatch: PublicDispatch) -> None:
    kwargs: Final[Mapping[str, object]] = {"custom_llm_provider": "openai", "dimensions": 8, "api_key": "sk-test"}

    request: Final = dispatch.request(("text-embedding-3-small", "hello"), kwargs)

    assert request is not None
    assert request.bound["model"] == "text-embedding-3-small"
    assert request.bound["input"] == "hello"
    assert request.bound["dimensions"] == 8
    assert request.bound["api_key"] == "sk-test"
    assert request.kwargs is kwargs
    assert dispatch.context(request) == RouteContext(
        Route.EMBEDDINGS, provider="openai", model="text-embedding-3-small"
    )


@pytest.mark.parametrize(
    ("args", "kwargs"),
    (
        pytest.param((None, "hello"), {}, id="unnamed-model"),
        pytest.param(("text-embedding-3-small", "hello"), {"model": "duplicate"}, id="does-not-bind"),
        pytest.param(("text-embedding-3-small", "hello"), {"aembedding": True}, id="internal-async-hop"),
    ),
)
def test_request_stays_on_python(args: tuple[object, ...], kwargs: Mapping[str, object]) -> None:
    assert _DISPATCH.request(args, kwargs) is None


def test_public_embedding_routes_through_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Final[list[NativeCall]] = []
    expected: Final = EmbeddingResponse(model="text-embedding-3-small", data=[])

    def native(call: NativeCall) -> EmbeddingResponse:
        captured.append(call)
        return expected

    NATIVE_EMBEDDING.override(native)
    monkeypatch.setattr(catalog, "decide", required_everywhere)
    public_embedding: Final = cast(Callable[..., EmbeddingResponse], litellm.embedding)
    try:
        result: Final = public_embedding(model="text-embedding-3-small", input="hello")
    finally:
        NATIVE_EMBEDDING.reset()
    assert result is expected
    assert [request.bound["model"] for request in captured] == ["text-embedding-3-small"]


@pytest.mark.asyncio
async def test_public_aembedding_routes_through_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: Final[list[NativeCall]] = []
    expected: Final = EmbeddingResponse(model="text-embedding-3-small", data=[])

    async def native(call: NativeCall) -> EmbeddingResponse:
        captured.append(call)
        return expected

    NATIVE_AEMBEDDING.override(native)
    monkeypatch.setattr(catalog, "decide", required_everywhere)
    public_aembedding: Final = cast(Callable[..., Awaitable[EmbeddingResponse]], litellm.aembedding)
    try:
        result: Final = await public_aembedding(model="text-embedding-3-small", input="hello")
    finally:
        NATIVE_AEMBEDDING.reset()
    assert result is expected
    assert [request.bound["model"] for request in captured] == ["text-embedding-3-small"]


@pytest.mark.asyncio
async def test_public_embedding_calls_keep_the_python_result() -> None:
    vector: Final = [0.1, 0.2]

    sync_response: Final = litellm.embedding(model="openai/test-model", input="hello", mock_response=vector)
    async_response: Final = await litellm.aembedding(model="openai/test-model", input="hello", mock_response=vector)

    assert isinstance(sync_response, EmbeddingResponse)
    rows: Final = TypeAdapter(list[dict[str, object]])
    assert rows.validate_python(sync_response.model_dump()["data"])[0]["embedding"] == vector
    assert rows.validate_python(async_response.model_dump()["data"])[0]["embedding"] == vector
