from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Final

import pytest
from pydantic import TypeAdapter

import litellm
from litellm.embeddings.dispatch import AEMBEDDING, EMBEDDING
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Route, RouteRule, Rules
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.public_call import NativeCall
from litellm.types.utils import EmbeddingResponse


@pytest.mark.asyncio
async def test_public_embedding_calls_keep_the_python_result() -> None:
    vector: Final = [0.1, 0.2]

    sync_response: Final = litellm.embedding(model="openai/test-model", input="hello", mock_response=vector)
    async_response: Final = await litellm.aembedding(model="openai/test-model", input="hello", mock_response=vector)

    assert isinstance(sync_response, EmbeddingResponse)
    rows: Final = TypeAdapter(list[dict[str, object]])
    assert rows.validate_python(sync_response.model_dump()["data"])[0]["embedding"] == vector
    assert rows.validate_python(async_response.model_dump()["data"])[0]["embedding"] == vector


def test_sync_embedding_request_projects_public_arguments() -> None:
    rules: Final[Rules] = (RouteRule(Route.EMBEDDINGS, Rollout.RUST_REQUIRED),)
    expected: Final = EmbeddingResponse(model="test-model", data=[])

    def native(request: NativeCall) -> EmbeddingResponse:
        assert request.resolved["model"] == "test-model"
        assert request.resolved["input"] == "hello"
        assert request.resolved["custom_llm_provider"] == "openai"
        return expected

    binding: Final[NativeBinding[Callable[[NativeCall], EmbeddingResponse]]] = NativeBinding(
        "embedding", validate=lambda _: None
    )
    binding.override(native)

    def python(*args: object, **kwargs: object) -> EmbeddingResponse:  # kwargs-ok: public pass-through shape
        pytest.fail("required native route must handle this call")

    response: Final = replace(EMBEDDING, binding=binding, python=python).run(
        ("test-model", "hello"),
        {"custom_llm_provider": "openai", "dimensions": 8},
        rules=rules,
    )

    assert response is expected


@pytest.mark.asyncio
async def test_async_embedding_falls_back_after_native_declines() -> None:
    from litellm.rust_bridge.bindings import native_exception_types

    native_types: Final = native_exception_types()
    if native_types is None:
        pytest.skip("native bridge is unavailable")
    declined, _ = native_types
    expected: Final = EmbeddingResponse(model="test-model", data=[])
    rules: Final[Rules] = (RouteRule(Route.EMBEDDINGS, Rollout.RUST_OPT_OUT),)

    async def native(request: NativeCall) -> EmbeddingResponse:
        raise declined("unsupported")

    async def python(*args: object, **kwargs: object) -> EmbeddingResponse:
        return expected

    binding: Final[NativeBinding[Callable[[NativeCall], Awaitable[EmbeddingResponse]]]] = NativeBinding(
        "aembedding", validate=lambda _: None
    )
    binding.override(native)
    response: Final = await replace(AEMBEDDING, binding=binding, python=python).arun(
        ("test-model", "hello"),
        {},
        rules=rules,
    )

    assert response is expected
