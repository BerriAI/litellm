"""Runs the auth prefetch's raw SQL against a real Postgres: the join must bind the membership to the requested
team and hand the getters rows they validate. The per-regime round-trip counts are unit-tested with fakes in
tests/test_litellm/proxy/auth/test_auth_object_prefetch.py."""

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.proxy.auth.auth_checks import (
    get_org_object,
    get_team_membership,
    get_team_object,
    get_user_object,
)
from litellm.proxy.auth.auth_object_prefetch import AuthObjectRefs, prefetch_auth_objects
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

pytestmark = pytest.mark.asyncio(loop_scope="session")


def _dead_db() -> MagicMock:
    prisma = MagicMock(name="prisma_client")
    prisma.db.query_first = AsyncMock(return_value=None)
    return prisma


async def test_join_binds_the_membership_to_the_requested_team(prisma):
    """A user in two teams with different member budgets must get the requested team's row."""
    run = uuid4().hex
    user_id, team_a, team_b, org_id = (f"pf-user-{run}", f"pf-team-a-{run}", f"pf-team-b-{run}", f"pf-org-{run}")
    try:
        await prisma.db.litellm_budgettable.create(
            data={"budget_id": f"a-{run}", "max_budget": 11.0, "created_by": "t", "updated_by": "t"}
        )
        await prisma.db.litellm_budgettable.create(
            data={"budget_id": f"b-{run}", "max_budget": 22.0, "created_by": "t", "updated_by": "t"}
        )
        await prisma.db.litellm_organizationtable.create(
            data={
                "organization_id": org_id,
                "organization_alias": "pf",
                "created_by": "t",
                "updated_by": "t",
                "litellm_budget_table": {"connect": {"budget_id": f"b-{run}"}},
            }
        )
        await prisma.db.litellm_usertable.create(data={"user_id": user_id, "max_budget": 33.0})
        await prisma.db.litellm_teamtable.create(data={"team_id": team_a, "organization_id": org_id, "max_budget": 1.0})
        await prisma.db.litellm_teamtable.create(data={"team_id": team_b, "max_budget": 2.0})
        await prisma.db.litellm_teammembership.create(
            data={"user_id": user_id, "team_id": team_a, "litellm_budget_table": {"connect": {"budget_id": f"a-{run}"}}}
        )
        await prisma.db.litellm_teammembership.create(
            data={"user_id": user_id, "team_id": team_b, "litellm_budget_table": {"connect": {"budget_id": f"b-{run}"}}}
        )

        cache = UserApiKeyCache(in_memory_cache=InMemoryCache(), redis_cache=None)
        refs = AuthObjectRefs(user_id=user_id, team_id=team_a, membership_user_id=user_id, organization_id=org_id)
        await prefetch_auth_objects(refs=refs, user_api_key_cache=cache, prisma_client=prisma)

        dead_db = _dead_db()
        membership = await get_team_membership(
            user_id=user_id, team_id=team_a, prisma_client=dead_db, user_api_key_cache=cache
        )
        team = await get_team_object(team_id=team_a, prisma_client=dead_db, user_api_key_cache=cache)
        user = await get_user_object(
            user_id=user_id, prisma_client=dead_db, user_api_key_cache=cache, user_id_upsert=False
        )
        org = await get_org_object(org_id=org_id, prisma_client=dead_db, user_api_key_cache=cache)
        assert dead_db.db.mock_calls == [], "getters must be served from the prefetched cache"

        assert membership is not None and membership.litellm_budget_table is not None
        assert (membership.team_id, membership.litellm_budget_table.max_budget) == (team_a, 11.0)
        assert (team.team_id, team.max_budget, team.organization_id, team.models) == (team_a, 1.0, org_id, [])
        assert user is not None and user.max_budget == 33.0
        assert org is not None and (org.organization_id, org.models) == (org_id, [])
    finally:
        await prisma.db.litellm_teammembership.delete_many(where={"user_id": user_id})
        await prisma.db.litellm_teamtable.delete_many(where={"team_id": {"in": [team_a, team_b]}})
        await prisma.db.litellm_usertable.delete_many(where={"user_id": user_id})
        await prisma.db.litellm_organizationtable.delete_many(where={"organization_id": org_id})
        await prisma.db.litellm_budgettable.delete_many(where={"budget_id": {"in": [f"a-{run}", f"b-{run}"]}})
