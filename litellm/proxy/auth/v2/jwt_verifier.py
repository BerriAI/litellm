import time
from typing import Any, Dict, Optional

from authlib.jose import JsonWebKey, jwt
from authlib.jose.errors import JoseError


class JWTVerificationError(Exception):
    """Raised when a token fails signature or standard-claim validation."""


def build_claims_options(issuer: Optional[str], audience: Optional[str]) -> Dict[str, Any]:
    options: Dict[str, Any] = {"exp": {"essential": True}}
    if issuer:
        options["iss"] = {"essential": True, "value": issuer}
    if audience:
        options["aud"] = {"essential": True, "value": audience}
    return options


def verify(
    token: str,
    key_set: Any,
    issuer: Optional[str] = None,
    audience: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify ``token`` against ``key_set`` and validate exp/iss/aud.

    authlib owns the crypto: signature verification, key selection by ``kid``,
    and standard-claim checks. ``key_set`` is an imported JWKS (injected so this
    is testable without network). Raises :class:`JWTVerificationError` on any
    failure so callers never branch on authlib's internal exception types.
    """
    try:
        claims = jwt.decode(
            token, key_set, claims_options=build_claims_options(issuer, audience)
        )
        claims.validate(now=int(time.time()))
        return dict(claims)
    except JoseError as e:
        raise JWTVerificationError(str(e)) from e
    except Exception as e:  # malformed token, bad header, etc.
        raise JWTVerificationError(f"invalid token: {e}") from e


class JWKSProvider:
    """Fetches and TTL-caches a JWKS document, returning an imported key set."""

    def __init__(self, jwks_uri: str, ttl_seconds: float = 600.0):
        self.jwks_uri = jwks_uri
        self.ttl_seconds = ttl_seconds
        self._key_set: Any = None
        self._fetched_at: float = 0.0

    async def get_key_set(self) -> Any:
        now = time.monotonic()
        if self._key_set is not None and (now - self._fetched_at) < self.ttl_seconds:
            return self._key_set

        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(self.jwks_uri)
            response.raise_for_status()
            self._key_set = JsonWebKey.import_key_set(response.json())
            self._fetched_at = now
        return self._key_set
