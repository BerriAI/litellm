"""The client_credentials (M2M) grant, isolated behind a port.

`resolve()` owns the orchestration (cache, expiry, fail closed); the RFC 6749 client-credentials
exchange is delegated here so it is not hand-rolled. The production body is backed by the MCP
SDK's `ClientCredentialsOAuthProvider`; a fake is injected in tests. Async network I/O.
"""

from __future__ import annotations

from typing import Protocol

from ..result import Result
from .token_store import StoredToken
from .types import ClientCredentialsConfig, CredError


class ClientCredentialsFetcher(Protocol):
    """Runs the client_credentials grant for a service account, or fails closed.

    Returns `misconfigured` when the grant is rejected (bad client_id / secret / scope, an
    operator error with no user to re-authenticate) and `upstream_unavailable` when the token
    endpoint cannot be reached.
    """

    async def fetch(
        self, config: ClientCredentialsConfig
    ) -> Result[StoredToken, CredError]: ...
