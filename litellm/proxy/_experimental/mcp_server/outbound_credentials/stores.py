"""Concrete credential sources behind the resolver's store seams.

A source does the fetch only; caching is layered separately (e.g. ``CachedByokStore``). These
read real storage and so import the surrounding package's data access; they are kept out of
the package ``__init__`` so the resolver core (``resolver.py`` / ``types.py`` / ``seams.py``)
stays import-clean.
"""

from __future__ import annotations

from typing import Optional


class DbBackedByokStore:
    """Reads the per-user BYOK key from the MCP credentials table via ``get_user_credential``.

    This is the persisted source of record (the BYOK entry flow writes it), not a temporary v1
    crutch, so it stays after the resolution migration. If the storage ever changes, a
    different source can slot in behind the ``ByokCredentialStore`` seam without touching the
    cache wrapper or the resolver arm.
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
