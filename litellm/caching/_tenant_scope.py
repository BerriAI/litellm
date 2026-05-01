"""Tenant-scope extraction for cache backends (VERIA-54).

The standard cache backends (Redis/in-memory/S3) hash the proxy-injected
metadata into the cache key, so two callers from different teams produce
different keys and cannot cross-read entries. Semantic caches operate on
prompt embeddings rather than the cache key, so they have no implicit
isolation — a caller from team B can read team A's cached response just
by sending a similar prompt.

This helper extracts a stable scope identifier from the same metadata
fields the standard cache uses. Both semantic backends key their storage
and retrieval on this scope so cross-tenant lookups become impossible.
"""

from typing import Any, Dict, Optional

# Canonical order so two callers with the same (team, user, org) tuple
# always produce the same scope string.
_TENANT_FIELDS = (
    "user_api_key_team_id",
    "user_api_key_user_id",
    "user_api_key_org_id",
)


def get_tenant_scope(kwargs: Dict[str, Any]) -> Optional[str]:
    """Return a stable scope identifier from proxy-injected metadata.

    Returns ``None`` when no scope is present (master key, direct SDK use,
    no proxy), in which case callers should fall back to the legacy
    shared cache pool — preserving BC for non-multi-tenant deployments.
    """
    metadata = kwargs.get("metadata") or {}
    if not isinstance(metadata, dict):
        return None
    parts = []
    for field in _TENANT_FIELDS:
        value = metadata.get(field)
        if isinstance(value, str) and value:
            parts.append(value)
    if not parts:
        return None
    return "|".join(parts)
