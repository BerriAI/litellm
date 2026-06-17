"""The per-subject credential store port for the `api_key` per-user / BYOK source.

`resolve()` owns the *pull* (which key, fail-closed on a miss); this is only the storage
mechanics behind a `Protocol`, so the resolver stays testable without a database. The
durable body (over `LiteLLM_MCPUserEnvVars` and the BYOK credential table) lands later;
`InMemoryCredentialStore` is a working body for tests and local wiring.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel, ConfigDict


class CredentialKey(BaseModel):
    """Identifies a per-user secret. Per-tenant / per-user isolation is the key shape."""

    model_config = ConfigDict(frozen=True)
    tenant_id: str
    subject_id: str
    server_id: str


class CredentialStore(Protocol):
    """Fetches the per-subject secret for an `api_key` per-user / BYOK server."""

    def get(self, key: CredentialKey) -> str | None: ...


class InMemoryCredentialStore:
    """A working in-memory `CredentialStore` for tests and local wiring."""

    def __init__(self, seeded: dict[CredentialKey, str] | None = None) -> None:
        self._values: dict[CredentialKey, str] = dict(seeded or {})

    def get(self, key: CredentialKey) -> str | None:
        return self._values.get(key)
