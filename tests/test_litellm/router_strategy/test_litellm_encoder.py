"""Tests for litellm/router_strategy/auto_router/litellm_encoder.py"""

from typing import Any, Final

import pytest


import litellm
from litellm.constants import DEFAULT_AUTO_ROUTER_MAX_INPUT_CHARS
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
        litellm_router_instance=router,  # pyright: ignore[reportArgumentType]  # test double stands in for Router
        model_name="text-embedding-3-small",
        **kwargs,
    )


class TestSendsDocsWholeUnlessAskedNotTo:
    """Cutting is opt-in: a caller that must see the whole text is never cut behind its back.

    The semantic guard and the MCP tool filter build this encoder without a limit. If a default
    limit ever creeps back in, a prompt-injection payload placed after a benign opener would be
    invisible to the guard while the full message still reaches the model.
    """

    def test_should_send_a_long_doc_whole_when_no_cap_is_configured(self):
        router: Final = RecordingRouter()
        long_doc: Final = "x" * 50_000

        _encoder(router).encode_queries([long_doc])

        assert router.embedded_inputs == [[long_doc]]

    @pytest.mark.asyncio
    async def test_should_send_a_long_doc_whole_when_no_cap_is_configured_async(self):
        router: Final = RecordingRouter()
        long_doc: Final = "x" * 50_000

        await _encoder(router).aencode_queries([long_doc])

        assert router.embedded_inputs == [[long_doc]]

    def test_should_send_a_long_doc_whole_when_the_cap_is_not_positive(self):
        router: Final = RecordingRouter()
        long_doc: Final = "x" * 50_000

        _encoder(router, max_input_chars=0).encode_queries([long_doc])

        assert router.embedded_inputs == [[long_doc]]


class TestEmbeddingInputCap:
    """With a cap configured, an embedding model's context window cannot fail the caller's request."""

    def test_should_cut_a_long_doc_to_the_cap_on_sync_encode(self):
        router: Final = RecordingRouter()

        _encoder(router, max_input_chars=DEFAULT_AUTO_ROUTER_MAX_INPUT_CHARS).encode_queries(["x" * 50_000])

        assert router.embedded_inputs == [["x" * DEFAULT_AUTO_ROUTER_MAX_INPUT_CHARS]]

    @pytest.mark.asyncio
    async def test_should_cut_a_long_doc_to_the_cap_on_async_encode(self):
        router: Final = RecordingRouter()

        await _encoder(router, max_input_chars=DEFAULT_AUTO_ROUTER_MAX_INPUT_CHARS).aencode_queries(["x" * 50_000])

        assert router.embedded_inputs == [["x" * DEFAULT_AUTO_ROUTER_MAX_INPUT_CHARS]]

    def test_should_cut_documents_as_well_as_queries(self):
        router: Final = RecordingRouter()

        _encoder(router, max_input_chars=100).encode_documents(["y" * 9_000])

        assert router.embedded_inputs == [["y" * 100]]

    @pytest.mark.asyncio
    async def test_should_cut_documents_as_well_as_queries_async(self):
        router: Final = RecordingRouter()

        await _encoder(router, max_input_chars=100).aencode_documents(["y" * 9_000])

        assert router.embedded_inputs == [["y" * 100]]

    def test_should_leave_docs_within_the_cap_untouched(self):
        router: Final = RecordingRouter()
        docs: Final = ["a short prompt", "b" * 100]

        _encoder(router, max_input_chars=100).encode_queries(docs)

        assert router.embedded_inputs == [docs]

    def test_should_cut_each_doc_in_a_batch_independently(self):
        router: Final = RecordingRouter()

        _encoder(router, max_input_chars=100).encode_queries(["short", "z" * 5_000])

        assert router.embedded_inputs == [["short", "z" * 100]]

    def test_should_keep_the_head_of_the_doc(self):
        router: Final = RecordingRouter()

        _encoder(router, max_input_chars=20).encode_queries(["route me please, then a huge pasted file"])

        assert router.embedded_inputs == [["route me please, the"]]
