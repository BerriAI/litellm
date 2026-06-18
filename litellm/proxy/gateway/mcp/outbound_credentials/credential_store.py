"""The per-subject credential store port for the `api_key` per-user / BYOK source.

`resolve()` owns the *pull* (which key, fail-closed on a miss); this is only the storage
mechanics behind a `Protocol`, so the resolver stays testable without a database. The
durable body (over `LiteLLM_MCPUserEnvVars` and the BYOK credential table) lands later;
`InMemoryCredentialStore` is a working body for tests and local wiring.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict

from ..result import Ok, Result
from .types import CredError


class CredentialKey(BaseModel):
    """Identifies a per-user secret. Per-tenant / per-user isolation is the key shape."""

    model_config = ConfigDict(frozen=True)
    tenant_id: str
    subject_id: str
    server_id: str


class CredentialStore(Protocol):
    """Fetches the per-subject secret for an `api_key` per-user / BYOK server.

    Returns a `Result` so a store/DB outage (`Error(upstream_unavailable)`) is distinct from a
    genuine miss (`Ok(None)`); the resolver maps the former to 503 and the latter to a 401/412.
    Async because the durable body queries Prisma / Redis on LiteLLM's async stack; a
    synchronous read would block the event loop. Both are settled now, before any caller, so
    the signature does not break when that body lands.
    """

    async def get(self, key: CredentialKey) -> Result[str | None, CredError]: ...


class InMemoryCredentialStore:
    """A working in-memory `CredentialStore` for tests and local wiring."""

    def __init__(self, seeded: dict[CredentialKey, str] | None = None) -> None:
        self._values: dict[CredentialKey, str] = dict(seeded or {})

    async def get(self, key: CredentialKey) -> Result[str | None, CredError]:
        return Ok(self._values.get(key))
