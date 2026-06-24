"""Injected ports the resolver depends on for the modes that fetch a credential.

These are pure contracts with no v1 imports, so the resolver core stays free of v1. The
concrete bodies (e.g. ``stores.py``) are injected at the composition root.
"""

from __future__ import annotations

from typing import Optional, Protocol


class ByokStoreUnavailable(Exception):
    """Raised by ``fetch`` when the backing store is unreachable (e.g. the DB is down).

    This is distinct from returning ``None`` for "the user has no key": a read-through cache
    skips caching the failure (so it is not pinned as "no key" for the TTL), and the caller maps
    it to its own fail-closed status (the override gate to 503, the resolver to its 401 challenge)
    rather than treating an outage as a definite absence.
    """


class ByokCredentialStore(Protocol):
    """Per-user (bring-your-own-key) credential lookup for the ``api_key`` BYOK source.

    Returns the user's stored key for an upstream, or ``None`` when they have not provisioned
    one (the arm turns that into a 401 challenge). The ``(user_id, server_id)`` pair fully
    scopes the lookup, so an implementation must never return one subject's key to another.
    The value is transient (fetched per request and handed straight to ``StaticHeaderAuth``,
    which holds it as a ``SecretStr``), so it is returned as a plain ``str``. Raises
    ``ByokStoreUnavailable`` when the backing store is unreachable, so an outage is never cached
    or read as a definite "no key".
    """

    async def fetch(self, user_id: str, server_id: str) -> Optional[str]: ...
