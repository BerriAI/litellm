"""Tests for litellm/router_strategy/auto_router/litellm_encoder.py"""

import os
import sys
from typing import Any, Final

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.constants import DEFAULT_MAX_EMBEDDING_INPUT_CHARS
from litellm.router_strategy.auto_router.litellm_encoder import LiteLLMRouterEncoder


class RecordingRouter:
    """Stand-in for the LiteLLM Router that records what reached the embedding call.

    Injected through the encoder's constructor so the assertion is on the real code path
    rather than on a patched attribute.
    """

    def __init__(self) -> None:
        self.embedded_inputs: list[list[str]] = []

    def _response(self, input: list[str]) -> litellm.EmbeddingResponse:
        self.embedded_inputs.append(list(input))
        return litellm.EmbeddingResponse(
            data=[{"embedding": [0.1, 0.2], "index": i, "object": "embedding"} for i in range(len(input))]
        )

    def embedding(self, input: list[str], model: str, **kwargs: Any) -> litellm.EmbeddingResponse:
        return self._response(input)

    async def aembedding(self, input: list[str], model: str, **kwargs: Any) -> litellm.EmbeddingResponse:
        return self._response(input)


def _encoder(router: RecordingRouter, **kwargs: Any) -> LiteLLMRouterEncoder:
    return LiteLLMRouterEncoder(
        litellm_router_instance=router,  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]  # test double stands in for Router
        model_name="text-embedding-3-small",
        **kwargs,
    )


class TestEmbeddingInputCap:
    """An embedding model's context window must not become the router's context window."""

    def test_should_cut_long_doc_to_default_cap_on_sync_encode(self):
        router: Final = RecordingRouter()
        long_doc: Final = "x" * 50_000

        _encoder(router).encode_queries([long_doc])

        assert router.embedded_inputs == [["x" * DEFAULT_MAX_EMBEDDING_INPUT_CHARS]]

    @pytest.mark.asyncio
    async def test_should_cut_long_doc_to_default_cap_on_async_encode(self):
        router: Final = RecordingRouter()
        long_doc: Final = "x" * 50_000

        await _encoder(router).aencode_queries([long_doc])

        assert router.embedded_inputs == [["x" * DEFAULT_MAX_EMBEDDING_INPUT_CHARS]]

    def test_should_cut_documents_as_well_as_queries(self):
        router: Final = RecordingRouter()

        _encoder(router).encode_documents(["y" * 9_000])

        assert router.embedded_inputs == [["y" * DEFAULT_MAX_EMBEDDING_INPUT_CHARS]]

    @pytest.mark.asyncio
    async def test_should_cut_documents_as_well_as_queries_async(self):
        router: Final = RecordingRouter()

        await _encoder(router).aencode_documents(["y" * 9_000])

        assert router.embedded_inputs == [["y" * DEFAULT_MAX_EMBEDDING_INPUT_CHARS]]

    def test_should_leave_docs_within_the_cap_untouched(self):
        router: Final = RecordingRouter()
        docs: Final = ["a short prompt", "b" * DEFAULT_MAX_EMBEDDING_INPUT_CHARS]

        _encoder(router).encode_queries(docs)

        assert router.embedded_inputs == [docs]

    def test_should_cut_each_doc_independently(self):
        router: Final = RecordingRouter()

        _encoder(router).encode_queries(["short", "z" * 5_000])

        assert router.embedded_inputs == [["short", "z" * DEFAULT_MAX_EMBEDDING_INPUT_CHARS]]

    def test_should_honor_a_configured_cap(self):
        router: Final = RecordingRouter()

        _encoder(router, max_input_chars=10).encode_queries(["0123456789abcdef"])

        assert router.embedded_inputs == [["0123456789"]]

    def test_should_disable_cutting_when_cap_is_not_positive(self):
        router: Final = RecordingRouter()
        long_doc: Final = "x" * 50_000

        _encoder(router, max_input_chars=0).encode_queries([long_doc])

        assert router.embedded_inputs == [[long_doc]]

    def test_should_keep_the_head_of_the_doc(self):
        router: Final = RecordingRouter()

        _encoder(router, max_input_chars=20).encode_queries(["route me please, then a huge pasted file"])

        assert router.embedded_inputs == [["route me please, the"]]
