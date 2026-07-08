"""
Regression tests for LiteLLMRouterEncoder embedding batching.

Large MCP tool catalogs (semantic tool filter) and auto-router route sets can
produce more documents than an embedding provider accepts in a single request
(e.g. OpenAI 2048, DeepInfra 1024). The encoder must split the input into
batches no larger than ``embedding_batch_size`` and concatenate the results in
order instead of sending everything in one oversized call.
"""

from typing import List, Optional

import pytest

import litellm
from litellm.router_strategy.auto_router.litellm_encoder import LiteLLMRouterEncoder


class _RecordingRouter:
    """Fake Router that records embedding call inputs and echoes a deterministic
    embedding per document so ordering can be asserted."""

    def __init__(self) -> None:
        self.sync_batches: List[List[str]] = []
        self.async_batches: List[List[str]] = []

    @staticmethod
    def _response(docs: List[str]) -> litellm.EmbeddingResponse:
        return litellm.EmbeddingResponse(
            model="m",
            data=[
                {"embedding": [float(len(doc))], "index": i, "object": "embedding"}
                for i, doc in enumerate(docs)
            ],
        )

    def embedding(self, input: List[str], model: str, **kwargs) -> litellm.EmbeddingResponse:
        self.sync_batches.append(list(input))
        return self._response(input)

    async def aembedding(self, input: List[str], model: str, **kwargs) -> litellm.EmbeddingResponse:
        self.async_batches.append(list(input))
        return self._response(input)


def _make_encoder(router: _RecordingRouter, batch_size: Optional[int] = None) -> LiteLLMRouterEncoder:
    kwargs = {} if batch_size is None else {"embedding_batch_size": batch_size}
    return LiteLLMRouterEncoder(
        litellm_router_instance=router,  # type: ignore[arg-type]
        model_name="openai/text-embedding-3-small",
        **kwargs,
    )


def test_encode_documents_splits_into_batches():
    router = _RecordingRouter()
    encoder = _make_encoder(router, batch_size=3)
    docs = [f"doc-{i}" for i in range(7)]

    result = encoder.encode_documents(docs)

    # 7 docs, batch size 3 -> batches of 3, 3, 1
    assert [len(b) for b in router.sync_batches] == [3, 3, 1]
    assert all(len(b) <= 3 for b in router.sync_batches)
    # concatenation preserves order: one embedding per doc, in input order
    assert result == [[float(len(d))] for d in docs]


def test_encode_documents_single_batch_when_under_limit():
    router = _RecordingRouter()
    encoder = _make_encoder(router, batch_size=100)
    docs = [f"doc-{i}" for i in range(10)]

    encoder.encode_documents(docs)

    assert len(router.sync_batches) == 1
    assert router.sync_batches[0] == docs


@pytest.mark.asyncio
async def test_aencode_documents_splits_into_batches():
    router = _RecordingRouter()
    encoder = _make_encoder(router, batch_size=2)
    docs = [f"doc-{i}" for i in range(5)]

    result = await encoder.aencode_documents(docs)

    assert [len(b) for b in router.async_batches] == [2, 2, 1]
    assert result == [[float(len(d))] for d in docs]


def test_call_uses_batching():
    router = _RecordingRouter()
    encoder = _make_encoder(router, batch_size=4)
    docs = [f"doc-{i}" for i in range(9)]

    result = encoder(docs)

    assert all(len(b) <= 4 for b in router.sync_batches)
    assert sum(len(b) for b in router.sync_batches) == 9
    assert result == [[float(len(d))] for d in docs]


def test_invalid_batch_size_falls_back_to_default():
    from litellm.constants import DEFAULT_EMBEDDING_ENCODER_BATCH_SIZE

    router = _RecordingRouter()
    encoder = _make_encoder(router, batch_size=0)

    assert encoder.embedding_batch_size == DEFAULT_EMBEDDING_ENCODER_BATCH_SIZE
