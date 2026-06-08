"""JWT principal extraction.

Decodes claims without verifying the signature (PyJWT direct). Signature
verification and DB-backed claim mapping live in ``JWTHandler.auth_builder``;
that path runs from the resolver when DB access is available.
"""

from typing import Optional

import jwt

from litellm.identity.jwt import parse_jwt_scopes
from litellm.identity.principal import JWTPrincipal


def extract_jwt_principal(token: Optional[str]) -> Optional[JWTPrincipal]:
    """Decode JWT claims without verification and build a ``JWTPrincipal``.

    Returns ``None`` when the token is missing or not JWT-shaped. The
    caller is responsible for invoking ``auth_builder`` (or another
    verifier) before trusting the principal for authorization.
    """
    if not token:
        return None

    try:
        jwt.get_unverified_header(token)
        claims = jwt.decode(token, options={"verify_signature": False})
    except jwt.PyJWTError:
        return None

    aud = claims.get("aud")

    return JWTPrincipal(
        sub=claims.get("sub"),
        iss=claims.get("iss"),
        aud=tuple(aud) if isinstance(aud, list) else aud,
        scopes=parse_jwt_scopes(claims),
        claims=claims,
    )
