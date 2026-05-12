"""
Authorization gate for caller-supplied tags.

A tag is "privileged" if it is wired to one of:
  - a deployment's ``litellm_params.tags`` (used by tag-based routing), or
  - a row in ``LiteLLM_TagTable`` with a non-null ``budget_id`` (used by
    tag-budget enforcement).

For privileged tags, callers must be authorized via a glob pattern in
``key.metadata.tags`` or ``team_metadata.tags``. Non-privileged tags pass
through unchanged; the caller's analytics tags (the common case) are
unaffected.

The privileged-tag set is cached in memory with a short TTL so the hot
path remains I/O-free. The cache reads from ``llm_router.model_list``
(already in memory) and one indexed scan of ``LiteLLM_TagTable``.
"""

import asyncio
import fnmatch
import time
from typing import Any, FrozenSet, Iterable, List, Optional

# How long the privileged-tag snapshot is considered fresh. Changes to
# deployments or the tag table take effect within this window.
_CACHE_TTL_SECONDS = 30.0

_cache_built_at: float = 0.0
_privileged_tags: FrozenSet[str] = frozenset()
_cache_lock = asyncio.Lock()


async def _rebuild_cache() -> None:
    """Snapshot the current privileged-tag set from router + tag table."""
    global _cache_built_at, _privileged_tags

    new_set: set = set()

    # Deferred imports to keep the proxy boot graph clean — this module is
    # imported by router code that is itself imported during proxy_server
    # construction.
    try:
        from litellm.proxy.proxy_server import llm_router
    except Exception:
        llm_router = None
    if llm_router is not None:
        for deployment in llm_router.model_list or []:
            params = deployment.get("litellm_params") or {}
            for t in params.get("tags") or []:
                if isinstance(t, str):
                    new_set.add(t)

    try:
        from litellm.proxy.proxy_server import prisma_client
    except Exception:
        prisma_client = None
    if prisma_client is not None and getattr(prisma_client, "db", None) is not None:
        try:
            rows = await prisma_client.db.litellm_tagtable.find_many(
                where={"budget_id": {"not": None}}
            )
            for r in rows:
                tag_name = getattr(r, "tag_name", None)
                if isinstance(tag_name, str):
                    new_set.add(tag_name)
        except Exception:
            # Never fail the request path on a cache rebuild error. The worst
            # case is a stale (or empty) snapshot — tags pass through.
            pass

    _privileged_tags = frozenset(new_set)
    _cache_built_at = time.time()


async def ensure_fresh_privileged_tags() -> None:
    """Rebuild the privileged-tag snapshot if the TTL has expired."""
    if time.time() - _cache_built_at <= _CACHE_TTL_SECONDS:
        return
    async with _cache_lock:
        if time.time() - _cache_built_at <= _CACHE_TTL_SECONDS:
            return
        await _rebuild_cache()


def get_privileged_tags_snapshot() -> FrozenSet[str]:
    """Return the most recent cached snapshot. Safe to call from sync code."""
    return _privileged_tags


def caller_authorized_for_tag(
    tag: str,
    *metadata_sources: Optional[Any],
) -> bool:
    """
    Return True iff ``tag`` matches any ``fnmatch`` pattern listed under
    ``tags`` in any of the provided metadata sources.

    Each source may be a dict (key/team/project metadata) or None.
    Sources that aren't dicts or don't carry a list under ``tags`` are
    skipped silently.
    """
    for meta in metadata_sources:
        if not isinstance(meta, dict):
            continue
        patterns = meta.get("tags") or []
        if not isinstance(patterns, list):
            continue
        for pattern in patterns:
            if isinstance(pattern, str) and fnmatch.fnmatchcase(tag, pattern):
                return True
    return False


def filter_authorized_tags(
    tags: Optional[Iterable[str]],
    privileged_tags: FrozenSet[str],
    *metadata_sources: Optional[Any],
) -> List[str]:
    """
    Return the subset of ``tags`` the caller is allowed to use.

    A tag survives the filter iff either:
      - it is not in ``privileged_tags`` (passes through unchanged), or
      - the caller has a glob pattern in one of the metadata sources that
        matches the tag.
    """
    if not tags:
        return []
    out: List[str] = []
    for tag in tags:
        if not isinstance(tag, str):
            continue
        if tag not in privileged_tags:
            out.append(tag)
        elif caller_authorized_for_tag(tag, *metadata_sources):
            out.append(tag)
    return out
