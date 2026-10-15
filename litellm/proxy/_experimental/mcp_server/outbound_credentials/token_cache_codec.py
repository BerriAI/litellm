"""Serialize + encrypt boundary for caching an OAuth token in a shared (Redis) cache.

Shared cache values contain an encrypted access token and optional identity-binding proof.
Refresh tokens remain in the database; cache TTL bounds the access token's lifetime.
Legacy bearer-only entries decode without proof and cannot satisfy identity enforcement.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
)

_BOUND_PREFIX: Final = "litellm-bound-oauth-v1:"
_BOUND_PAYLOAD: Final = TypeAdapter(dict[str, str])


@dataclass(frozen=True, slots=True)
class OAuthTokenCacheCodec:
    encrypt: Callable[[str], str]
    decrypt: Callable[[str], str | None]

    def encode(self, token: OAuthToken) -> str:
        if token.identity_binding_proof is not None:
            return self.encrypt(
                _BOUND_PREFIX
                + json.dumps(
                    {
                        "access_token": token.access_token,
                        "identity_binding_proof": token.identity_binding_proof,
                    }
                )
            )
        return self.encrypt(token.access_token)

    def decode(self, blob: str) -> OAuthToken | None:
        access_token: Final = self.decrypt(blob)
        if not access_token:
            return None
        if access_token.startswith(_BOUND_PREFIX):
            try:
                payload: Final = _BOUND_PAYLOAD.validate_json(access_token[len(_BOUND_PREFIX) :])
            except ValidationError:
                return None
            bearer: Final = payload.get("access_token")
            proof: Final = payload.get("identity_binding_proof")
            if not bearer or not proof:
                return None
            return OAuthToken(access_token=bearer, identity_binding_proof=proof)
        return OAuthToken(access_token=access_token, refresh_token=None)
