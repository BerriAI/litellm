"""OAuth 2.0 bearer-token validation (resource-server role).

The MCP server is a pure resource server: it never issues tokens. It validates
the signed JWT access tokens that an external authorization server issued to
the calling AI agent, checking signature (against the issuer's JWKS), issuer,
audience, expiry, and a required scope.

This module intentionally has no dependency on the ``mcp`` package so the
validation logic can be unit-tested in isolation; the thin adapter that maps a
``VerifiedToken`` onto ``mcp``'s ``AccessToken`` lives in ``server.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jwt
from jwt import PyJWKClient


class TokenValidationError(Exception):
    """Raised when a presented bearer token is missing, malformed, or invalid."""


@dataclass(frozen=True)
class VerifiedToken:
    subject: str
    client_id: str
    scopes: list[str] = field(default_factory=list)
    expires_at: int | None = None


def _extract_scopes(claims: dict) -> list[str]:
    """Collect scopes across the claim shapes used by common IdPs.

    OAuth standard ``scope`` is a space-delimited string; Azure AD uses ``scp``
    (string) and ``roles`` (list); Auth0 uses ``permissions`` (list).
    """
    scopes: set[str] = set()
    for key in ("scope", "scp"):
        value = claims.get(key)
        if isinstance(value, str):
            scopes.update(value.split())
        elif isinstance(value, list):
            scopes.update(str(item) for item in value)
    for key in ("roles", "permissions"):
        value = claims.get(key)
        if isinstance(value, list):
            scopes.update(str(item) for item in value)
    return sorted(scopes)


class JWKSValidator:
    """Validates JWT access tokens against an issuer's published JWKS.

    The underlying ``PyJWKClient`` fetches the JWKS lazily and caches signing
    keys, so steady-state validation does no network IO. ``validate`` is
    blocking (JWKS refresh + crypto); callers on an event loop should run it in
    a worker thread.
    """

    def __init__(
        self,
        *,
        jwks_url: str,
        issuer: str,
        audience: str,
        algorithms: tuple[str, ...],
        required_scope: str | None = None,
    ) -> None:
        self._issuer = issuer
        self._audience = audience
        self._algorithms = list(algorithms)
        self._required_scope = required_scope
        self._jwk_client = PyJWKClient(jwks_url, cache_keys=True, lifespan=600)

    def validate(self, token: str) -> VerifiedToken:
        if not token:
            raise TokenValidationError("empty bearer token")

        try:
            signing_key = self._jwk_client.get_signing_key_from_jwt(token)
        except Exception as exc:  # PyJWK raises several distinct error types
            raise TokenValidationError(f"unable to resolve signing key: {exc}") from exc

        try:
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=self._algorithms,
                audience=self._audience,
                issuer=self._issuer,
                options={"require": ["exp", "iss", "aud"]},
            )
        except jwt.PyJWTError as exc:
            raise TokenValidationError(f"token rejected: {exc}") from exc

        scopes = _extract_scopes(claims)
        if self._required_scope and self._required_scope not in scopes:
            raise TokenValidationError(
                f"token is missing required scope '{self._required_scope}'"
            )

        return VerifiedToken(
            subject=str(claims.get("sub", "")),
            client_id=str(
                claims.get("client_id") or claims.get("azp") or claims.get("sub", "")
            ),
            scopes=scopes,
            expires_at=claims.get("exp"),
        )
