from dataclasses import dataclass, field
from typing import Any, Dict, Optional


class JWTClaimError(Exception):
    """Raised when verified claims lack the data needed to form an identity."""


@dataclass
class JWTSettings:
    jwks_uri: str
    issuer: Optional[str] = None
    audience: Optional[str] = None
    user_id_claim: str = "sub"
    team_claim: Optional[str] = None
    role_claim: Optional[str] = None
    # Maps an upstream role/group value to a litellm role name. Unmapped values
    # yield no role rather than trusting an arbitrary IdP string as a role.
    role_map: Dict[str, str] = field(default_factory=dict)


@dataclass
class JWTIdentity:
    user_id: str
    team_id: Optional[str] = None
    role: Optional[str] = None


def extract_identity(claims: Dict[str, Any], settings: JWTSettings) -> JWTIdentity:
    """Map verified claims to an identity using the configured claim names.

    The signature is already trusted at this point; this is the LiteLLM-semantic
    layer authlib does not own.
    """
    user_id = claims.get(settings.user_id_claim)
    if not user_id:
        raise JWTClaimError(f"token missing user id claim '{settings.user_id_claim}'")

    team_id = claims.get(settings.team_claim) if settings.team_claim else None

    role: Optional[str] = None
    if settings.role_claim:
        raw_role = claims.get(settings.role_claim)
        if raw_role is not None:
            role = settings.role_map.get(raw_role)

    return JWTIdentity(
        user_id=str(user_id),
        team_id=str(team_id) if team_id is not None else None,
        role=role,
    )
