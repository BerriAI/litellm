"""JWT principal extraction.

Uses ``JWTHandler.is_jwt`` for shape detection and
``JWTHandler.get_unverified_claims`` for claim peek. Signature
verification and DB-backed claim mapping live in ``JWTHandler.auth_builder``;
that path runs from the resolver when DB access is available.
"""

from typing import Optional

from litellm.identity.principal import JWTPrincipal


def extract_jwt_principal(token: Optional[str]) -> Optional[JWTPrincipal]:
    """Decode JWT claims without verification and build a ``JWTPrincipal``.

    Returns ``None`` when the token is missing or not JWT-shaped. The
    caller is responsible for invoking ``auth_builder`` (or another
    verifier) before trusting the principal for authorization.
    """
    if not token:
        return None

    from litellm.proxy.auth.handle_jwt import JWTHandler

    if not JWTHandler.is_jwt(token=token):
        return None

    claims = JWTHandler.get_unverified_claims(token=token) or {}

    aud = claims.get("aud")
    scope_claim = claims.get("scope") or claims.get("scp") or ""
    if isinstance(scope_claim, list):
        scopes = [str(s) for s in scope_claim if s]
    elif isinstance(scope_claim, str):
        scopes = [s for s in scope_claim.split(" ") if s]
    else:
        scopes = []

    return JWTPrincipal(
        sub=claims.get("sub"),
        iss=claims.get("iss"),
        aud=aud,
        scopes=scopes,
        claims=claims,
    )
