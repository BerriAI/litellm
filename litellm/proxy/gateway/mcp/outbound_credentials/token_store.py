"""The per-subject OAuth token store for the `authorization_code` arm.

Holds the user's stored upstream token (access + optional refresh + expiry), keyed by
`(tenant, subject, server, resource)` so tokens are per-user and audience-bound (RFC 8707).
The AS surface writes it during the OAuth dance; `resolve()` reads it. Async because the
durable body queries Prisma / Redis on LiteLLM's async stack; `InMemoryTokenStore` is a
working body for tests and local wiring.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, SecretStr

from ..result import Ok, Result
from .types import CredError


class TokenKey(BaseModel):
    """Identifies a per-user upstream token. Per-tenant / per-user / per-audience isolation."""

    model_config = ConfigDict(frozen=True)
    tenant_id: str
    subject_id: str
    server_id: str
    resource: str  # RFC 8707 audience the token is bound to


class StoredToken(BaseModel):
    """A user's upstream OAuth token as persisted. Secrets are `SecretStr` so they never log."""

    model_config = ConfigDict(frozen=True)
    access_token: SecretStr
    expires_at: datetime
    refresh_token: SecretStr | None = None


class TokenStore(Protocol):
    """Persists and retrieves per-`(subject, server, resource)` OAuth tokens.

    Both methods return a `Result` so a store/DB outage (`Error(upstream_unavailable)`, 503)
    is distinct from a genuine miss (`Ok(None)`, which drives the OAuth dance). Async because
    the durable body queries Prisma / Redis on LiteLLM's async stack.
    """

    async def get(self, key: TokenKey) -> Result[StoredToken | None, CredError]: ...

    async def put(
        self, key: TokenKey, token: StoredToken
    ) -> Result[None, CredError]: ...


class InMemoryTokenStore:
    """A working in-memory `TokenStore` for tests and local wiring."""

    def __init__(self, seeded: dict[TokenKey, StoredToken] | None = None) -> None:
        self._tokens: dict[TokenKey, StoredToken] = dict(seeded or {})

    async def get(self, key: TokenKey) -> Result[StoredToken | None, CredError]:
        return Ok(self._tokens.get(key))

    async def put(self, key: TokenKey, token: StoredToken) -> Result[None, CredError]:
        self._tokens[key] = (
            token  # mutable-ok: an in-memory store's backing must be mutable
        )
        return Ok(None)
