"""One-shot ``OAuthTokenStore`` for the create/test tools preview.

The preview tests an unsaved server, so no per-user credential is persisted yet. The operator holds
the just-authorized token; this serves it through the same v2 resolver path runtime uses for the
stored token, so the preview never relies on the caller-credential-override path that
``_create_mcp_client`` refuses for ``authorization_code``. It backs a single preview call, so it
returns its one token regardless of the lookup key.
"""

from __future__ import annotations

from dataclasses import dataclass

from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
)


@dataclass(frozen=True, slots=True)
class PresentedOAuthTokenStore:
    """Serves one in-hand token for the single preview call it backs (no DB, no cache)."""

    token: OAuthToken

    async def fetch(self, user_id: str, server_id: str) -> OAuthToken | None:
        return self.token
