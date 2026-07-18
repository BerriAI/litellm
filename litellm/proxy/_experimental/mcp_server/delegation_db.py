from datetime import datetime, timezone

from litellm.proxy.common_utils.user_api_key_cache import get_management_object_ttl
from litellm.proxy.utils import PrismaClient
from litellm.repositories.table_repositories import UserAgentDelegationRepository
from litellm.types.mcp_server.user_agent_delegation import UserAgentDelegation

# Sentinel cached for a (user, agent) pair with no active consent, so a repeated
# unauthorized assertion is answered from cache instead of re-querying the DB.
_NO_ACTIVE_DELEGATION = "__no_active_delegation__"


def _delegation_cache_key(user_id: str, agent_id: str) -> str:
    return f"mcp_user_delegation:{user_id}:{agent_id}"


async def grant_user_agent_delegation(
    prisma_client: PrismaClient,
    user_id: str,
    agent_id: str,
    granted_by: str,
) -> UserAgentDelegation:
    """Grant consent for `agent_id` to act on behalf of `user_id`.

    Upserts so re-granting after a revocation reactivates the same row
    (revoked_at cleared) instead of violating the (user_id, agent_id) unique.
    """
    row = await UserAgentDelegationRepository(prisma_client).table.upsert(
        where={"user_id_agent_id": {"user_id": user_id, "agent_id": agent_id}},
        data={
            "create": {"user_id": user_id, "agent_id": agent_id, "granted_by": granted_by},
            "update": {
                "granted_at": datetime.now(timezone.utc),
                "granted_by": granted_by,
                "revoked_at": None,
                "revoked_by": None,
            },
        },
    )
    await _invalidate_delegation_cache(user_id, agent_id)
    return UserAgentDelegation(**row.model_dump())


async def revoke_user_agent_delegation(
    prisma_client: PrismaClient,
    user_id: str,
    agent_id: str,
    revoked_by: str,
) -> UserAgentDelegation | None:
    """Revoke an ACTIVE consent. Returns None when there is nothing active to
    revoke (no record, or already revoked), so the caller surfaces a 404 and a
    repeat revoke never overwrites the original revoked_at/revoked_by audit."""
    existing = await UserAgentDelegationRepository(prisma_client).table.find_unique(
        where={"user_id_agent_id": {"user_id": user_id, "agent_id": agent_id}}
    )
    if existing is None or existing.revoked_at is not None:
        return None
    row = await UserAgentDelegationRepository(prisma_client).table.update(
        where={"user_id_agent_id": {"user_id": user_id, "agent_id": agent_id}},
        data={"revoked_at": datetime.now(timezone.utc), "revoked_by": revoked_by},
    )
    await _invalidate_delegation_cache(user_id, agent_id)
    return UserAgentDelegation(**row.model_dump()) if row else None


async def get_active_user_agent_delegation(
    prisma_client: PrismaClient,
    user_id: str,
    agent_id: str,
) -> UserAgentDelegation | None:
    """The active (never-revoked) consent row for this pair, if any.

    Cached in ``user_api_key_cache`` (positive and negative) so a delegated
    request does not hit the DB on the hot path; grant and revoke both bust the
    entry, so a consent change still takes effect immediately across workers
    rather than only at TTL.
    """
    from litellm.proxy.proxy_server import user_api_key_cache

    cache_key = _delegation_cache_key(user_id, agent_id)
    cached = await user_api_key_cache.async_get_cache(key=cache_key)
    if cached == _NO_ACTIVE_DELEGATION:
        return None
    if isinstance(cached, dict):
        return UserAgentDelegation(**cached)

    row = await UserAgentDelegationRepository(prisma_client).table.find_first(
        where={"user_id": user_id, "agent_id": agent_id, "revoked_at": None}
    )
    ttl = get_management_object_ttl(user_api_key_cache)
    if row is None:
        await user_api_key_cache.async_set_cache(key=cache_key, value=_NO_ACTIVE_DELEGATION, ttl=ttl)
        return None
    delegation = UserAgentDelegation(**row.model_dump())
    await user_api_key_cache.async_set_cache(key=cache_key, value=delegation.model_dump(mode="json"), ttl=ttl)
    return delegation


async def list_user_agent_delegations(
    prisma_client: PrismaClient,
    user_id: str | None = None,
) -> list[UserAgentDelegation]:
    """List delegations, scoped to `user_id` when given.

    `user_id=None` means "all rows across all users" and is an ADMIN-ONLY view;
    callers gating on a non-admin identity must pass that identity's user_id (and
    handle a None identity themselves) rather than falling through to this
    unscoped listing.
    """
    where = {"user_id": user_id} if user_id is not None else {}
    rows = await UserAgentDelegationRepository(prisma_client).table.find_many(where=where, order={"granted_at": "desc"})
    return [UserAgentDelegation(**r.model_dump()) for r in rows]


async def _invalidate_delegation_cache(user_id: str, agent_id: str) -> None:
    from litellm.proxy.proxy_server import user_api_key_cache

    await user_api_key_cache.async_delete_cache(key=_delegation_cache_key(user_id, agent_id))
