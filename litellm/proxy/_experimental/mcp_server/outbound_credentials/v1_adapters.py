"""v1-backed bodies for the resolver's injected seams.

The lean approach: orchestrate, do not reimplement. Each body delegates to v1's existing
credential machinery, so these import v1 and are wired in only at the composition root. They
are kept out of the package ``__init__`` so the resolver core (``resolver.py`` / ``types.py``
/ ``seams.py``) stays free of v1 imports.
"""

from __future__ import annotations

from typing import Optional


class V1BackedByokStore:
    """``ByokCredentialStore`` *source* backed by v1's persisted per-user credential lookup.

    Reads the stored key through v1's ``get_user_credential`` and nothing more. Caching is a
    separate concern layered by ``CachedByokStore``, which wraps this. Step 1b swaps this for a
    v2-native source (reading v2-owned storage) behind the same seam, leaving the cache wrapper
    unchanged.
    """

    async def fetch(self, user_id: str, server_id: str) -> Optional[str]:
        if not user_id:
            return None

        from litellm.proxy._experimental.mcp_server.db import get_user_credential
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            return None

        credential = await get_user_credential(
            prisma_client=prisma_client, user_id=user_id, server_id=server_id
        )
        return credential or None
