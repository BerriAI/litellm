"""The OAuth refresh-token grant, isolated behind a port.

`resolve()` owns the orchestration (which token, expiry decision, persist, fail closed); the
actual RFC 6749 refresh exchange is delegated here so it is not hand-rolled. The production
body is backed by the MCP SDK / a standard OAuth client; a fake is injected in tests. Async
network I/O, so it lands behind this Protocol with its real body wired later.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import SecretStr

from ..result import Result
from .token_store import StoredToken
from .types import AuthorizationCodeConfig, CredError


class TokenRefresher(Protocol):
    """Exchanges a refresh token for a fresh `StoredToken`, or fails closed.

    Returns `unauthorized` when the grant is rejected (refresh token revoked/expired, the user
    must re-authenticate) and `upstream_unavailable` when the token endpoint cannot be reached.
    """

    async def refresh(
        self, config: AuthorizationCodeConfig, refresh_token: SecretStr
    ) -> Result[StoredToken, CredError]: ...
