"""Identity-cache invalidation hooks.

Two flavors:

- Per-token: a key was rotated, blocked, or deleted. We know the exact
  token hash, so we drop the entry from both memory and Redis.

- Per-scope (team / user / org): a row that fans out to many keys
  changed. Rather than enumerating every key that references the team,
  we bump a generation counter for that scope. Cached identities carry
  the scope generations they were minted under; reads compare and treat
  a mismatch as a miss.

The legacy ``_delete_cache_key_object`` and the per-table cache deletes
in ``auth_checks.py`` stay in place. These hooks run side-by-side so we
don't strand a partially-deployed fleet that's still reading from the
legacy cache keys.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from litellm.identity.cache import (
    IdentityCache,
    org_generation_key,
    team_generation_key,
    user_generation_key,
)

if TYPE_CHECKING:
    from litellm.caching.dual_cache import DualCache


def _identity_cache_for(dual_cache: "DualCache") -> IdentityCache:
    return IdentityCache(dual_cache=dual_cache)


async def invalidate_identity_for_token(
    *, token_hash: str, dual_cache: "DualCache"
) -> None:
    """Drop a single key's cached identity."""
    await _identity_cache_for(dual_cache).delete(token_hash)


async def invalidate_identity_for_team(
    *, team_id: str, dual_cache: "DualCache"
) -> None:
    """Mark every identity that references this team as stale."""
    await _identity_cache_for(dual_cache).bump_generation(
        team_generation_key(team_id)
    )


async def invalidate_identity_for_user(
    *, user_id: str, dual_cache: "DualCache"
) -> None:
    """Mark every identity that references this user as stale."""
    await _identity_cache_for(dual_cache).bump_generation(
        user_generation_key(user_id)
    )


async def invalidate_identity_for_org(
    *, org_id: str, dual_cache: "DualCache"
) -> None:
    """Mark every identity that references this organization as stale."""
    await _identity_cache_for(dual_cache).bump_generation(
        org_generation_key(org_id)
    )
