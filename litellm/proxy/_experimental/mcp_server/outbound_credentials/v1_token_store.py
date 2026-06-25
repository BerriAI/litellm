"""v1-backed ``OAuthTokenStore`` source for the ``authorization_code`` mode.

Resolves the user's access token through v1's egress core (``resolve_user_oauth_access_token``:
Redis cache, else DB read with refresh), so the v2 arm injects exactly the token v1 would, with the
same silent refresh. This is the strangler adapter; step 1b swaps the core for a v2-native store
behind the same ``OAuthTokenStore`` seam. It imports v1, so it is kept out of the package
``__init__`` like the rest of the adapter layer.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
)

if TYPE_CHECKING:
    from litellm.types.mcp_server.mcp_server_manager import MCPServer


class V1PerUserTokenStore:
    """``OAuthTokenStore`` backed by v1's per-user OAuth egress.

    ``fetch`` looks the server up by id (injected ``server_lookup``) and resolves the token through
    v1's ``resolve_user_oauth_access_token``, which refreshes an expired token when a refresh_token
    is stored and re-warms the Redis cache. Returns ``None`` when the user has no usable token (the
    arm turns that into a challenge); v1's core swallows store errors as a miss, so this never
    raises ``TokenStoreUnavailable``. The returned ``OAuthToken`` carries only the access token —
    refresh happens inside the core, not via ``RefreshingTokenStore``.
    """

    def __init__(self, server_lookup: Callable[[str], MCPServer | None]) -> None:
        self._server_lookup = server_lookup

    async def fetch(self, user_id: str, server_id: str) -> OAuthToken | None:
        if not user_id:
            return None
        server = self._server_lookup(server_id)
        if server is None:
            return None

        from litellm.proxy._experimental.mcp_server.db import (
            resolve_user_oauth_access_token,
        )

        access_token = await resolve_user_oauth_access_token(user_id, server)
        return OAuthToken(access_token=access_token) if access_token else None
