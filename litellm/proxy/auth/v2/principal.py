from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth


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


def _role_to_str(role: object) -> Optional[str]:
    if role is None:
        return None
    if isinstance(role, Enum):
        value = role.value
        return value if isinstance(value, str) else str(value)
    return str(role)


def build_principal(identity: UserAPIKeyAuth) -> Principal:
    """Derive a :class:`Principal` from an authenticated identity."""
    user_id = identity.user_id
    team_id = identity.team_id
    token = identity.token

    if user_id:
        subject = f"user:{user_id}"
    elif token:
        subject = f"key:{token}"
    else:
        subject = "anonymous"

    domain = f"team:{team_id}" if team_id else "*"

    groupings: List[List[str]] = []
    role = _role_to_str(identity.user_role)
    if role:
        groupings.append([subject, f"role:{role}"])

    return Principal(subject=subject, domain=domain, groupings=groupings)
