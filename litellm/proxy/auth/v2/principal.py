from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass(frozen=True)
class Principal:
    """The casbin-facing view of an authenticated identity.

    ``subject`` and ``domain`` are the request coordinates; ``groupings`` are the
    ``g`` rows bridging this identity to its casbin roles, derived from identity
    data that already exists (the key's ``user_role``). New decision logic,
    existing identity data.
    """

    subject: str
    domain: str
    groupings: List[List[str]]


def _role_to_str(role: Any) -> Optional[str]:
    if role is None:
        return None
    return getattr(role, "value", role)


def build_principal(identity: Any) -> Principal:
    """Derive a :class:`Principal` from an authenticated identity object.

    Duck-typed on ``user_id`` / ``team_id`` / ``token`` / ``user_role`` so it
    stays decoupled from the full ``UserAPIKeyAuth`` import.
    """
    user_id = getattr(identity, "user_id", None)
    team_id = getattr(identity, "team_id", None)
    token = getattr(identity, "token", None)

    if user_id:
        subject = f"user:{user_id}"
    elif token:
        subject = f"key:{token}"
    else:
        subject = "anonymous"

    domain = f"team:{team_id}" if team_id else "*"

    groupings: List[List[str]] = []
    role = _role_to_str(getattr(identity, "user_role", None))
    if role:
        groupings.append([subject, f"role:{role}"])

    return Principal(subject=subject, domain=domain, groupings=groupings)
