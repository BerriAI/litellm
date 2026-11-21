from __future__ import annotations

from collections.abc import Callable
from typing import Final

import pytest

from litellm.embeddings import dispatch
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.catalog import Route, RouteRule, Rules
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.public_call import NativeCall, native_call_hook
from litellm.types.utils import EmbeddingResponse


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
    response: Final = dispatch._DISPATCH.run(  # pyright: ignore[reportPrivateUsage]  # test an explicit route decision
        ("test-model", "hello"),
        {"custom_llm_provider": "openai", "dimensions": 8},
        python=lambda *args, **kwargs: pytest.fail("required native route must handle this call"),
        binding=binding,
        native=native_call_hook,
        rules=rules,
    )

    assert response is expected
