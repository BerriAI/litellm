"""Injected ports the resolver depends on for the modes that fetch a credential.

These are pure contracts with no v1 imports, so the resolver core stays free of v1. The
v1-backed bodies live in ``v1_adapters.py`` and are injected at the composition root.
"""

from __future__ import annotations

from typing import Optional, Protocol


class ByokCredentialStore(Protocol):
    """Per-user (bring-your-own-key) credential lookup for the ``api_key`` BYOK source.

    Returns the user's stored key for an upstream, or ``None`` when they have not provisioned
    one (the arm turns that into a 401 challenge). The ``(user_id, server_id)`` pair fully
    scopes the lookup, so an implementation must never return one subject's key to another.
    The value is transient (fetched per request and handed straight to ``StaticHeaderAuth``,
    which holds it as a ``SecretStr``), so it is returned as a plain ``str``.
    """

    async def fetch(self, user_id: str, server_id: str) -> Optional[str]: ...
