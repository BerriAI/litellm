from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Final

import pytest
from pydantic import TypeAdapter

import litellm
from litellm.embeddings import dispatch
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Route, RouteRule, Rules
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.embeddings.entrypoints import LiteLLMEmbeddingRequest
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

    def native(
        request: LiteLLMEmbeddingRequest, args: tuple[object, ...], kwargs: Mapping[str, object]
    ) -> EmbeddingResponse:
        assert request.model == "test-model"
        assert request.input == "hello"
        assert request.custom_llm_provider == "openai"
        return expected

    binding: Final[
        NativeBinding[Callable[[LiteLLMEmbeddingRequest, tuple[object, ...], Mapping[str, object]], EmbeddingResponse]]
    ] = NativeBinding("embedding", validate=lambda _: None)
    binding.override(native)
    response: Final = dispatch._DISPATCH.run(  # pyright: ignore[reportPrivateUsage]  # test an explicit route decision
        ("test-model", "hello"),
        {"custom_llm_provider": "openai", "dimensions": 8},
        python=lambda *args, **kwargs: pytest.fail("required native route must handle this call"),
        binding=binding,
        native=lambda hook, request, args, kwargs: hook(request, args, kwargs),
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

    async def native(
        request: LiteLLMEmbeddingRequest, args: tuple[object, ...], kwargs: Mapping[str, object]
    ) -> EmbeddingResponse:
        raise declined("unsupported")

    async def python(*args: object, **kwargs: object) -> EmbeddingResponse:
        return expected

    binding: Final[
        NativeBinding[
            Callable[[LiteLLMEmbeddingRequest, tuple[object, ...], Mapping[str, object]], Awaitable[EmbeddingResponse]]
        ]
    ] = NativeBinding("aembedding", validate=lambda _: None)
    binding.override(native)
    response: Final = await dispatch._ADISPATCH.arun(  # pyright: ignore[reportPrivateUsage]  # test an explicit route decision
        ("test-model", "hello"),
        {},
        python=python,
        binding=binding,
        native=lambda hook, request, args, kwargs: hook(request, args, kwargs),
        rules=rules,
    )

    assert response is expected
