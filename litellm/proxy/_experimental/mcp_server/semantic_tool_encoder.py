"""
Caching encoder for the shared MCP semantic tool index.

Lives apart from semantic_tool_index_store because LiteLLMRouterEncoder pulls
in semantic_router, an optional extra that must stay out of the store's import
chain.
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

from litellm.proxy._experimental.mcp_server.semantic_tool_index_store import (
    ToolVectorStore,
    tool_vector_key,
)
from litellm.router_strategy.auto_router.litellm_encoder import LiteLLMRouterEncoder

if TYPE_CHECKING:
    from litellm.router import Router


class CachingToolEncoder(LiteLLMRouterEncoder):
    """LiteLLMRouterEncoder that serves document embeddings from a ToolVectorStore.

    Queries are never cached; only add/aadd's document utterances go through
    the store. Duplicate texts share one key and one embed. Cached vectors
    whose length disagrees with a live probe are treated as misses, so a
    stale entry can never feed the index a wrong-shape vector.
    """

    _tool_vector_store: ToolVectorStore
    _embedding_identity: str
    _dimensions: int | None

    def __init__(
        self,
        *,
        store: ToolVectorStore,
        embedding_identity: str,
        litellm_router_instance: "Router",
        model_name: str,
        score_threshold: float | None = None,
    ) -> None:
        super().__init__(
            litellm_router_instance=litellm_router_instance,
            model_name=model_name,
            score_threshold=score_threshold,
        )
        self._tool_vector_store = store
        self._embedding_identity = embedding_identity
        self._dimensions = None

    async def adimensions(self) -> int:
        if self._dimensions is None:
            probe: Final = await self.aencode_queries(["test"])  # pyright: ignore[reportUnknownMemberType]  # semantic_router ships no type stubs
            self._dimensions = len(probe[0])
        return self._dimensions

    def encode_documents(  # pyright: ignore[reportIncompatibleMethodOverride]  # semantic_router calls encode_documents without kwargs
        self, docs: Sequence[str]
    ) -> list[list[float]]:  # mutable-ok: mirrors the parent DenseEncoder contract
        return super().encode_documents(list(docs))

    async def aencode_documents(  # pyright: ignore[reportIncompatibleMethodOverride]  # semantic_router calls aencode_documents without kwargs
        self, docs: Sequence[str]
    ) -> list[list[float]]:  # mutable-ok: mirrors the parent DenseEncoder contract
        unique_docs: Final = tuple(dict.fromkeys(docs))
        keys: Final = tuple(tool_vector_key(self._embedding_identity, text) for text in unique_docs)
        hits: Final[Mapping[str, tuple[float, ...]]] = await self._tool_vector_store.get_many(keys)
        valid_hits: Final = (
            {key: vector for key, vector in hits.items() if len(vector) == await self.adimensions()} if hits else {}
        )

        miss_pairs: Final = tuple((text, key) for text, key in zip(unique_docs, keys) if key not in valid_hits)
        embedded: Final = dict(valid_hits)
        if miss_pairs:
            vectors: Final = await super().aencode_documents([text for text, _ in miss_pairs])
            new_vectors: Final = {key: tuple(vector) for (_, key), vector in zip(miss_pairs, vectors)}
            await self._tool_vector_store.put_many(new_vectors)
            embedded.update(new_vectors)

        vectors_by_text: Final = dict(zip(unique_docs, (embedded[key] for key in keys)))
        return [list(vectors_by_text[text]) for text in docs]
