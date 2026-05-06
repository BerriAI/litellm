"""
Ownership / access-control helpers for the three-level agent hierarchy
(Agent -> Session -> Run).

Owner identity for an agent is the SHA-256 hash of the API key that created
it (``user_api_key_hash``). This matches the existing pattern used by
``litellm_managedobjecttable`` for containers — but we store the hash on
the row directly because there are no nested resource lookups here.

Sessions and runs inherit ownership from their parent agent — so if a
caller can read the parent, they can read the child. Cross-tenant isolation
is enforced uniformly by ``check_agent_ownership`` which is called from
every public endpoint before any read or write.
"""

from typing import Any, Optional

from fastapi import HTTPException

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy._types import hash_token as _hash_litellm_api_key


def caller_api_key_hash(user_api_key_dict: UserAPIKeyAuth) -> str:
    """Return the SHA-256 hash of the caller's API key.

    ``user_api_key_dict.api_key`` may already be hashed (DB-issued virtual
    keys store the hash), or it may be a master key in plain form. Either
    way we hash whatever we get — hashing an already-hashed value gives a
    deterministic value that can never collide with a real raw key.
    """
    api_key = user_api_key_dict.api_key
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key on auth context")
    return _hash_litellm_api_key(api_key)


def is_proxy_admin(user_api_key_dict: UserAPIKeyAuth) -> bool:
    """Proxy admins can access any agent / session / run regardless of owner."""
    role = user_api_key_dict.user_role
    if role is None:
        return False
    return role in {
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    }


def assert_caller_owns_agent(
    user_api_key_dict: UserAPIKeyAuth,
    agent_row: Any,
) -> None:
    """Raise 404 if the caller is not the agent's owner (and not an admin).

    404 (not 403) on purpose — leaking existence of another tenant's
    resource is a fingerprinting risk.
    """
    if agent_row is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    if is_proxy_admin(user_api_key_dict):
        return
    expected_hash = caller_api_key_hash(user_api_key_dict)
    if agent_row.user_api_key_hash != expected_hash:
        raise HTTPException(status_code=404, detail="Agent not found")


def assert_caller_owns_session(
    user_api_key_dict: UserAPIKeyAuth,
    session_row: Any,
) -> None:
    """Raise 404 if the caller is not the session's owner (and not admin).

    Sessions carry their own ``user_api_key_hash`` so we don't need to load
    the parent agent to check ownership."""
    if session_row is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if is_proxy_admin(user_api_key_dict):
        return
    expected_hash = caller_api_key_hash(user_api_key_dict)
    if session_row.user_api_key_hash != expected_hash:
        raise HTTPException(status_code=404, detail="Session not found")


def owner_filter_for_caller(
    user_api_key_dict: UserAPIKeyAuth,
) -> Optional[dict]:
    """Build a Prisma ``where`` clause restricting list queries to the
    caller's own rows.

    Returns ``None`` for proxy admins (no filter) so callers can spread it
    into an existing where dict only when needed.
    """
    if is_proxy_admin(user_api_key_dict):
        return None
    return {"user_api_key_hash": caller_api_key_hash(user_api_key_dict)}
