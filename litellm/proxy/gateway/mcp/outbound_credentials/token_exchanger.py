"""The RFC 8693 token-exchange (OBO) grant, isolated behind a port.

`resolve()` owns the orchestration (cache, expiry, fail closed, and sending the inbound token
ONLY here, never to the upstream); the exchange itself is delegated to this port. The
production body is a hand-rolled RFC 8693 POST over httpx (the SDK ships only the deprecated
RFC 7523 jwt-bearer grant), naming the gateway as `act` and requesting a token bound to
`resource` (RFC 8707) at the discovered token endpoint. Faked in tests.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import SecretStr

from ..result import Result
from .token_store import StoredToken
from .types import CredError, TokenExchangeConfig


class TokenExchanger(Protocol):
    """Swaps the caller's `subject_token` for a token bound to `resource`, or fails closed.

    Returns `unauthorized` when the subject_token is invalid/expired (the user must
    re-authenticate to the gateway), `misconfigured` when the exchange config is wrong (the
    endpoint or IdP does not support exchange), and `upstream_unavailable` when the endpoint is
    unreachable.
    """

    async def exchange(
        self, config: TokenExchangeConfig, subject_token: SecretStr, resource: str
    ) -> Result[StoredToken, CredError]: ...
