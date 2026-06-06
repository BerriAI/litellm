from typing import List, Optional

VALID_ACTIONS = {"read", "write", "delete", "manage", "call"}
VALID_EFFECTS = {"allow", "deny"}
VALID_SUBJECT_TYPES = {"user", "team"}


class PolicyValidationError(ValueError):
    """Raised when a policy/assignment request is malformed."""


def _role_token(role: str) -> str:
    if not role or not role.strip():
        raise PolicyValidationError("role is required")
    return role if role.startswith("role:") else f"role:{role}"


def normalize_object(resource: str, resource_id: Optional[str]) -> str:
    if not resource or not resource.strip():
        raise PolicyValidationError("resource is required")
    return f"{resource}:{resource_id}" if resource_id else f"{resource}:*"


def make_permission_rule(
    role: str,
    resource: str,
    action: str,
    effect: str = "allow",
    domain: str = "*",
    resource_id: Optional[str] = None,
) -> List[str]:
    """Build a casbin ``p`` rule row from validated, normalized inputs."""
    if action not in VALID_ACTIONS:
        raise PolicyValidationError(
            f"action must be one of {sorted(VALID_ACTIONS)}, got '{action}'"
        )
    if effect not in VALID_EFFECTS:
        raise PolicyValidationError(
            f"effect must be one of {sorted(VALID_EFFECTS)}, got '{effect}'"
        )
    obj = normalize_object(resource, resource_id)
    return ["p", _role_token(role), domain or "*", obj, action, effect]


def make_assignment_rule(
    subject_type: str,
    subject_id: str,
    role: str,
    domain: Optional[str] = None,
) -> List[str]:
    """Build a casbin role-assignment rule binding a subject to a role.

    A global assignment is a ``g`` row; a domain-scoped one (the role only
    applies within ``domain``, e.g. ``team:eng``) is a ``g3`` row.
    """
    if subject_type not in VALID_SUBJECT_TYPES:
        raise PolicyValidationError(
            f"subject_type must be one of {sorted(VALID_SUBJECT_TYPES)}, "
            f"got '{subject_type}'"
        )
    if not subject_id or not subject_id.strip():
        raise PolicyValidationError("subject_id is required")

    subject = f"{subject_type}:{subject_id}"
    role_token = _role_token(role)
    if domain and domain != "*":
        return ["g3", subject, role_token, domain]
    return ["g", subject, role_token]
