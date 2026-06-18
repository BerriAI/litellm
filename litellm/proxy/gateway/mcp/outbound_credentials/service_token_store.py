"""The shared service-account token cache for the `client_credentials` (M2M) arm.

The M2M token is the same for every user of a server, so it is keyed by `(server_id, resource)`
with no subject. It is a pure optimization cache, not a source of truth: the token is always
re-mintable from the client secret, so the resolver degrades gracefully on a cache outage
(re-mint on a read failure, best-effort on write) rather than failing. Redis-with-TTL later;
`InMemoryServiceTokenStore` is a working body for tests and local wiring.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict

from ..result import Ok, Result
from .token_store import StoredToken
from .types import CredError


class ServiceTokenKey(BaseModel):
    """Identifies a shared service-account token. No subject: one identity for all users."""

    model_config = ConfigDict(frozen=True)
    server_id: str
    resource: str  # RFC 8707 audience the token is bound to


class ServiceTokenStore(Protocol):
    """Caches the shared M2M token per `(server_id, resource)`."""

    async def get(
        self, key: ServiceTokenKey
    ) -> Result[StoredToken | None, CredError]: ...

    async def put(
        self, key: ServiceTokenKey, token: StoredToken
    ) -> Result[None, CredError]: ...


class InMemoryServiceTokenStore:
    """A working in-memory `ServiceTokenStore` for tests and local wiring."""

    def __init__(
        self, seeded: dict[ServiceTokenKey, StoredToken] | None = None
    ) -> None:
        self._tokens: dict[ServiceTokenKey, StoredToken] = dict(seeded or {})

    async def get(self, key: ServiceTokenKey) -> Result[StoredToken | None, CredError]:
        return Ok(self._tokens.get(key))

    async def put(
        self, key: ServiceTokenKey, token: StoredToken
    ) -> Result[None, CredError]:
        self._tokens[key] = (
            token  # mutable-ok: an in-memory store's backing must be mutable
        )
        return Ok(None)
