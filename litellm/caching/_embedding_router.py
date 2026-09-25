"""Shared selection of the embedding path for semantic caches.

Both the Redis and qdrant semantic caches need the same decision: when the
configured embedding model is a proxy Router deployment, embeddings must run
through the Router so per-deployment auth (e.g. Bedrock aws_role_name) is
applied. Otherwise fall back to a direct litellm embedding call.

This module is dependency-injected: callers pass the proxy ``llm_router`` and
``llm_model_list`` in, so the decision logic is unit-testable without importing
``litellm.proxy.proxy_server``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final

import litellm
from litellm.constants import SEMANTIC_CACHE_EMBEDDING_TIMEOUT_SECONDS

if TYPE_CHECKING:
    from litellm.router import Router


def resolve_embedding_router(
    embedding_model: str,
    llm_router: Router | None,
    llm_model_list: list[dict[str, Any]] | None,
) -> Router | None:
    """Return ``llm_router`` iff it serves ``embedding_model`` as a deployment."""
    if llm_router is None:
        return None
    router_model_names: Final[list[str]] = (
        [m["model_name"] for m in llm_model_list if "model_name" in m] if llm_model_list is not None else []
    )
    if embedding_model in router_model_names:
        return llm_router
    return None


def build_router_embedding_metadata(
    request_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    """Forward the caller's full metadata, flagged as a semantic-cache embedding."""
    metadata: Final[dict[str, Any]] = dict(request_metadata or {})
    metadata["semantic-cache-embedding"] = True
    return metadata


def resolve_embedding_max_input_tokens(
    configured_max_input_tokens: int | None,
    embedding_model: str,
    router: Router | None,
) -> int | None:
    """Explicit cache setting first, else the Router deployment's configured ``max_input_tokens``."""
    if configured_max_input_tokens is not None:
        return configured_max_input_tokens
    if router is None:
        return None
    deployment_max_input_tokens, _ = router.get_configured_token_limits(embedding_model)
    return deployment_max_input_tokens


def resolve_embedding_timeout(configured_timeout: float | None) -> float:
    """Explicit cache setting first, else the short semantic-cache default."""
    if configured_timeout is not None:
        return configured_timeout
    return SEMANTIC_CACHE_EMBEDDING_TIMEOUT_SECONDS


def truncate_embedding_input(prompt: str, embedding_model: str, max_input_tokens: int | None) -> str:
    """Keep only the first ``max_input_tokens`` tokens of ``prompt`` for the embedding call."""
    if max_input_tokens is None:
        return prompt
    tokens: Final[Sequence[int]] = litellm.encode(model=embedding_model, text=prompt)
    if len(tokens) <= max_input_tokens:
        return prompt
    truncated: Final[str] = litellm.decode(model=embedding_model, tokens=tokens[:max_input_tokens])
    return truncated
