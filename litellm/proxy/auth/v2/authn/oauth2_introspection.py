from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class OAuth2IntrospectionError(Exception):
    """Raised when an introspection response is inactive or unusable."""


@dataclass
class IntrospectionSettings:
    endpoint: str
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    user_id_claim: str = "sub"
    team_claim: Optional[str] = None
    scope_claim: str = "scope"
    # Maps an OAuth scope value to a litellm role name.
    role_map: Dict[str, str] = field(default_factory=dict)


@dataclass
class IntrospectionIdentity:
    user_id: str
    team_id: Optional[str] = None
    role: Optional[str] = None


def _scopes(raw: object) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return raw.split()
    if isinstance(raw, (list, tuple)):
        return [str(s) for s in raw]
    return []


def parse_introspection_response(
    data: Dict[str, Any], settings: IntrospectionSettings
) -> IntrospectionIdentity:
    """Validate an RFC 7662 introspection response and map it to an identity.

    Per the spec, ``active`` is the authoritative liveness flag; a token is only
    valid when it is explicitly active.
    """
    if data.get("active") is not True:
        raise OAuth2IntrospectionError("token is not active")

    user_id = data.get(settings.user_id_claim) or data.get("sub")
    if not user_id:
        raise OAuth2IntrospectionError(
            f"introspection response missing user id claim '{settings.user_id_claim}'"
        )

    team_id = data.get(settings.team_claim) if settings.team_claim else None

    role: Optional[str] = None
    for scope in _scopes(data.get(settings.scope_claim)):
        mapped = settings.role_map.get(scope)
        if mapped:
            role = mapped
            break

    return IntrospectionIdentity(
        user_id=str(user_id),
        team_id=str(team_id) if team_id is not None else None,
        role=role,
    )
