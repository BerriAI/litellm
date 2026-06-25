"""v1-backed ``OAuthTokenStore`` source for the ``authorization_code`` mode.

Reads the user's stored access token through v1's ``mcp_per_user_token_cache`` (a Redis-backed,
encrypted-at-rest per-user cache). This is a temporary adapter: step 1b replaces it with a
v2-native token store that also tracks expiry and refresh, behind the same ``OAuthTokenStore`` seam.
It imports v1, so it is kept out of the package ``__init__`` like the rest of the adapter layer.
"""

from __future__ import annotations

from typing import Optional

from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
)


class V1PerUserTokenStore:
    """``OAuthTokenStore`` backed by v1's per-user token cache.

    v1 stores only the access token (the cache TTL is its lifetime), so the ``OAuthToken`` carries
    no ``expires_at`` or ``refresh_token``: the v2 cache holds it for its default TTL, and the OAuth
    challenge drives re-auth once v1's cache drops it. v1's ``get`` swallows errors as a miss, so
    this never raises ``TokenStoreUnavailable``.
    """

    async def fetch(self, user_id: str, server_id: str) -> Optional[OAuthToken]:
        if not user_id:
            return None

        from litellm.proxy._experimental.mcp_server.oauth2_token_cache import (
            mcp_per_user_token_cache,
        )

        access_token = await mcp_per_user_token_cache.get(user_id, server_id)
        if not access_token:
            return None
        return OAuthToken(access_token=access_token)
