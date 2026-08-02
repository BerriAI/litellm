"""Shared selection of the embedding path for semantic caches.

Both the Redis and qdrant semantic caches need the same decision: when the
configured embedding model is a proxy Router deployment, embeddings must run
through the Router so per-deployment auth (e.g. Bedrock aws_role_name) is
applied. Otherwise fall back to a direct litellm embedding call.

This module is dependency-injected: callers pass the proxy ``llm_router`` in, so
the decision logic is unit-testable without importing
``litellm.proxy.proxy_server``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from litellm.router import Router


def resolve_embedding_router(
    embedding_model: str,
    llm_router: Router | None,
) -> Router | None:
    """Return ``llm_router`` iff it serves ``embedding_model`` as a deployment."""
    if llm_router is None:
        return None
    if llm_router.get_model_list(model_name=embedding_model):
        return llm_router
    return None


def build_router_embedding_metadata(
    request_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Forward the caller's full metadata, flagged as a semantic-cache embedding."""
    metadata: dict[str, Any] = dict(request_metadata or {})
    metadata["semantic-cache-embedding"] = True
    return metadata
