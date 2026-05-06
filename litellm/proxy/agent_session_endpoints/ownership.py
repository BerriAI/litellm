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
    """True iff the caller is a full proxy admin (read AND write).

    ``PROXY_ADMIN_VIEW_ONLY`` is intentionally NOT included here. For
    write paths, this function being False means the view-only admin
    falls back to the per-tenant ownership check, which they fail (no
    matching ``user_api_key_hash``) and get a 404 — closing the
    privilege escalation that previously let view-only admins create /
    update / delete other tenants' agents, sessions, and runs.

    For read paths, callers should use :func:`is_proxy_admin_read` so
    view-only admins still get cross-tenant visibility.
    """
    role = user_api_key_dict.user_role
    if role is None:
        return False
    return role == LitellmUserRoles.PROXY_ADMIN


def is_proxy_admin_read(user_api_key_dict: UserAPIKeyAuth) -> bool:
    """True iff the caller is any flavor of proxy admin (incl. view-only).

    Used only on read paths where the view-only admin is allowed to see
    other tenants' resources.
    """
    role = user_api_key_dict.user_role
    if role is None:
        return False
    return role in {
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
    }


def assert_caller_can_mutate(user_api_key_dict: UserAPIKeyAuth) -> None:
    """Reject write access for view-only admins.

    Every state-mutating endpoint (POST/PUT/PATCH/DELETE) must call this
    before any DB write. View-only admins are granted read access to all
    tenants via :func:`is_proxy_admin_read` but MUST NOT be allowed to
    mutate state — they otherwise inherit master-admin write authority
    over every tenant's agents, sessions, and runs.
    """
    if user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY:
        raise HTTPException(
            status_code=403,
            detail="View-only admins cannot perform write operations",
        )


def assert_caller_owns_agent(
    user_api_key_dict: UserAPIKeyAuth,
    agent_row: Any,
) -> None:
    """Raise 404 if the caller is not the agent's owner (and not an admin).

    404 (not 403) on purpose — leaking existence of another tenant's
    resource is a fingerprinting risk.

    Both full and view-only admins pass this read-side check; write
    endpoints must additionally call :func:`assert_caller_can_mutate` to
    block view-only admins from mutating other tenants' rows.
    """
    if agent_row is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    if is_proxy_admin_read(user_api_key_dict):
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
    the parent agent to check ownership.

    Both full and view-only admins pass this read-side check; write
    endpoints must additionally call :func:`assert_caller_can_mutate` to
    block view-only admins from mutating other tenants' rows.
    """
    if session_row is None:
        raise HTTPException(status_code=404, detail="Session not found")
    if is_proxy_admin_read(user_api_key_dict):
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
    into an existing where dict only when needed. Both full and view-only
    admins get the unfiltered read.
    """
    if is_proxy_admin_read(user_api_key_dict):
        return None
    return {"user_api_key_hash": caller_api_key_hash(user_api_key_dict)}
