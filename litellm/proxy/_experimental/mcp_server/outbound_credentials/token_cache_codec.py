"""Serialize + encrypt boundary for caching an OAuth token in a shared (Redis) cache.

A cross-replica cache must serialize the token, and a plaintext bearer in Redis is a leak, so this
encrypts the value (NaCl in production via the injected ``encrypt``, identity in tests). It caches
**only** the ``access_token``: the hot path needs just the bearer, expiry is carried by the cache
entry's TTL (set from the token's ``expires_at`` by the cache), and the long-lived refresh_token stays
in the DB - the refresh path is always a cache miss that re-reads it - so it never reaches Redis. A
decoded token therefore carries only the bearer (``expires_at`` and ``refresh_token`` both None); the
TTL, not the value, bounds its life. An empty/undecryptable blob (e.g. master-key rotation) is a miss.
"""

from __future__ import annotations

from collections.abc import Callable

from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
)


class OAuthTokenCacheCodec:
    def __init__(
        self,
        encrypt: Callable[[str], str],
        decrypt: Callable[[str], str | None],
    ) -> None:
        self._encrypt = encrypt
        self._decrypt = decrypt

    def encode(self, token: OAuthToken) -> str:
        return self._encrypt(token.access_token)

    def decode(self, blob: str) -> OAuthToken | None:
        access_token = self._decrypt(blob)
        if not access_token:
            return None
        return OAuthToken(access_token=access_token, refresh_token=None)
