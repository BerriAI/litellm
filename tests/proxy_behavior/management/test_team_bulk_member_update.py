import pytest

from .actors import Actor
from .conftest import create_scratch_team

pytestmark = pytest.mark.asyncio(loop_scope="session")

ROUTE = "/v2/team/{team_id}/members"


def _role_of(row, user_id: str):
    for m in row.members_with_roles or []:
        if m["user_id"] == user_id:
            return m["role"]
    return None


# PATCH /v2/team/{team_id}/members — same coarse route gate and admin checks as
# POST /team/member_update, so the actor x team-shape matrix mirrors it: the
# scratch team is raw-seeded with a "user"-role member and each scenario bulk
# promotes it to "admin". PROXY_ADMIN, the team's team admin, or an org admin of
# the team's org may update; everyone else is 403. (The harness forces
# premium_user, so the admin-role premium gate never decides the outcome.)
_MATRIX = [
    ("alpha/proxy_admin", Actor.PROXY_ADMIN, "alpha", 200),
    ("alpha/org_admin", Actor.ORG_ADMIN, "alpha", 200),
    ("alpha/team_admin", Actor.TEAM_ADMIN, "alpha", 200),
    ("alpha/internal_user", Actor.INTERNAL_USER, "alpha", 403),
    ("alpha/owner", Actor.OWNER, "alpha", 403),
    ("alpha/unrelated_same_org", Actor.UNRELATED_SAME_ORG, "alpha", 403),
    ("alpha/cross_org_user", Actor.CROSS_ORG_USER, "alpha", 403),
    ("alpha/service_account", Actor.SERVICE_ACCOUNT, "alpha", 403),
    ("alpha/org_b_admin", Actor.ORG_B_ADMIN, "alpha", 403),
    ("beta/proxy_admin", Actor.PROXY_ADMIN, "beta", 200),
    ("beta/org_admin", Actor.ORG_ADMIN, "beta", 403),
    ("beta/team_admin", Actor.TEAM_ADMIN, "beta", 403),
    ("beta/internal_user", Actor.INTERNAL_USER, "beta", 403),
    ("beta/owner", Actor.OWNER, "beta", 403),
    ("beta/unrelated_same_org", Actor.UNRELATED_SAME_ORG, "beta", 403),
    ("beta/cross_org_user", Actor.CROSS_ORG_USER, "beta", 403),
    ("beta/service_account", Actor.SERVICE_ACCOUNT, "beta", 403),
    ("beta/org_b_admin", Actor.ORG_B_ADMIN, "beta", 200),
]


async def _seed_target(prisma, world, shape: str, team_id: str, member_id: str) -> None:
    if shape == "alpha":
        await create_scratch_team(
            prisma,
            team_id,
            organization_id=world.org_a_id,
            admin_user_ids=[world.keys[Actor.TEAM_ADMIN].user_id],
            member_user_ids=[member_id],
        )
    elif shape == "beta":
        await create_scratch_team(
            prisma,
            team_id,
            organization_id=world.org_b_id,
            member_user_ids=[member_id],
        )
    else:  # pragma: no cover - guard
        pytest.fail(f"unknown shape={shape}")


@pytest.mark.parametrize(
    "actor,shape,expected_status",
    [(a, sh, s) for (_id, a, sh, s) in _MATRIX],
    ids=[s[0] for s in _MATRIX],
)
async def test_bulk_member_update_authz_matrix(
    actor: Actor,
    shape: str,
    expected_status: int,
    proxy_client,
    prisma,
    scratch,
    world,
):
    member_id = scratch.tag("member")
    await _seed_target(prisma, world, shape, scratch.prefix, member_id)
    caller = world.keys[actor]

    resp = await proxy_client.patch(
        ROUTE.format(team_id=scratch.prefix),
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={"user_ids": [member_id], "update_fields": {"role": "admin"}},
    )
    assert (
        resp.status_code == expected_status
    ), f"{actor.value} {shape}: {resp.status_code} {resp.text}"

    row = await prisma.db.litellm_teamtable.find_unique(
        where={"team_id": scratch.prefix}
    )
    assert row is not None
    if expected_status == 200:
        assert _role_of(row, member_id) == "admin"
    else:
        assert _role_of(row, member_id) == "user", "denied but role changed"


async def test_bulk_member_update_reports_non_members_without_failing_batch(
    proxy_client, prisma, scratch, world
):
    """Valid members are all promoted in one write; a user_id that is not a
    member of the team lands in failed_updates while the rest still succeed."""
    m1 = scratch.tag("m1")
    m2 = scratch.tag("m2")
    stranger = scratch.tag("stranger")
    await create_scratch_team(
        prisma,
        scratch.prefix,
        organization_id=world.org_a_id,
        member_user_ids=[m1, m2],
    )

    resp = await proxy_client.patch(
        ROUTE.format(team_id=scratch.prefix),
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={
            "user_ids": [m1, m2, stranger],
            "update_fields": {"role": "admin"},
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_requested"] == 3
    assert {u["user_id"] for u in body["successful_updates"]} == {m1, m2}
    assert [f["user_id"] for f in body["failed_updates"]] == [stranger]

    row = await prisma.db.litellm_teamtable.find_unique(
        where={"team_id": scratch.prefix}
    )
    assert row is not None
    assert _role_of(row, m1) == "admin"
    assert _role_of(row, m2) == "admin"


async def test_bulk_member_update_all_members_in_team_promotes_everyone(
    proxy_client, prisma, scratch, world
):
    """all_members_in_team=True applies the patch to every current member."""
    m1 = scratch.tag("m1")
    m2 = scratch.tag("m2")
    await create_scratch_team(
        prisma,
        scratch.prefix,
        organization_id=world.org_a_id,
        member_user_ids=[m1, m2],
    )

    resp = await proxy_client.patch(
        ROUTE.format(team_id=scratch.prefix),
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"all_members_in_team": True, "update_fields": {"role": "admin"}},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["total_requested"] == 2

    row = await prisma.db.litellm_teamtable.find_unique(
        where={"team_id": scratch.prefix}
    )
    assert row is not None
    assert _role_of(row, m1) == "admin"
    assert _role_of(row, m2) == "admin"


async def test_bulk_member_update_writes_member_budget_row(
    proxy_client, prisma, scratch, world
):
    """A limit patch on a member with no budget row creates one membership +
    budget carrying the patched value; this is the set-based write path."""
    member_id = scratch.tag("member")
    await create_scratch_team(
        prisma,
        scratch.prefix,
        organization_id=world.org_a_id,
        member_user_ids=[member_id],
    )

    resp = await proxy_client.patch(
        ROUTE.format(team_id=scratch.prefix),
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"user_ids": [member_id], "update_fields": {"tpm_limit": 4242}},
    )
    assert resp.status_code == 200, resp.text

    membership = await prisma.db.litellm_teammembership.find_unique(
        where={"user_id_team_id": {"user_id": member_id, "team_id": scratch.prefix}},
        include={"litellm_budget_table": True},
    )
    assert membership is not None and membership.litellm_budget_table is not None
    assert membership.litellm_budget_table.tpm_limit == 4242


async def test_bulk_member_update_clearing_last_limit_disconnects_private_budget(
    proxy_client, prisma, scratch, world
):
    member_id = scratch.tag("member")
    budget_id = scratch.tag("bud")
    await create_scratch_team(
        prisma,
        scratch.prefix,
        organization_id=world.org_a_id,
        member_user_ids=[member_id],
    )
    await prisma.db.litellm_budgettable.create(
        data={"budget_id": budget_id, "tpm_limit": 999, "created_by": "t", "updated_by": "t"}
    )
    await prisma.db.litellm_teammembership.create(
        data={"user_id": member_id, "team_id": scratch.prefix, "budget_id": budget_id}
    )

    resp = await proxy_client.patch(
        ROUTE.format(team_id=scratch.prefix),
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"user_ids": [member_id], "update_fields": {"tpm_limit": None}},
    )
    assert resp.status_code == 200, resp.text

    membership = await prisma.db.litellm_teammembership.find_unique(
        where={"user_id_team_id": {"user_id": member_id, "team_id": scratch.prefix}},
        include={"litellm_budget_table": True},
    )
    assert membership is not None
    assert membership.budget_id is None, "cleared budget must be disconnected"
    assert membership.litellm_budget_table is None


async def test_bulk_member_update_does_not_leak_default_limits_to_null_budget_members(
    proxy_client, prisma, scratch, world
):
    on_default = scratch.tag("ondefault")
    no_budget = scratch.tag("nobudget")
    default_budget_id = scratch.tag("defbud")
    await create_scratch_team(
        prisma,
        scratch.prefix,
        organization_id=world.org_a_id,
        member_user_ids=[on_default, no_budget],
        metadata={"team_member_budget_id": default_budget_id},
    )
    await prisma.db.litellm_budgettable.create(
        data={"budget_id": default_budget_id, "max_budget": 100.0, "created_by": "t", "updated_by": "t"}
    )
    await prisma.db.litellm_teammembership.create(
        data={"user_id": on_default, "team_id": scratch.prefix, "budget_id": default_budget_id}
    )
    await prisma.db.litellm_teammembership.create(
        data={"user_id": no_budget, "team_id": scratch.prefix}
    )

    resp = await proxy_client.patch(
        ROUTE.format(team_id=scratch.prefix),
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"user_ids": [on_default, no_budget], "update_fields": {"tpm_limit": 42}},
    )
    assert resp.status_code == 200, resp.text

    on_default_m = await prisma.db.litellm_teammembership.find_unique(
        where={"user_id_team_id": {"user_id": on_default, "team_id": scratch.prefix}},
        include={"litellm_budget_table": True},
    )
    assert on_default_m is not None and on_default_m.litellm_budget_table is not None
    assert on_default_m.budget_id != default_budget_id, "must clone, not patch the shared row"
    assert on_default_m.litellm_budget_table.tpm_limit == 42
    assert on_default_m.litellm_budget_table.max_budget == 100.0, "clone inherits the default's limits"

    no_budget_m = await prisma.db.litellm_teammembership.find_unique(
        where={"user_id_team_id": {"user_id": no_budget, "team_id": scratch.prefix}},
        include={"litellm_budget_table": True},
    )
    assert no_budget_m is not None and no_budget_m.litellm_budget_table is not None
    assert no_budget_m.litellm_budget_table.tpm_limit == 42
    assert no_budget_m.litellm_budget_table.max_budget is None, "null-budget member must not inherit default limits"

    default_row = await prisma.db.litellm_budgettable.find_unique(where={"budget_id": default_budget_id})
    assert default_row is not None and default_row.max_budget == 100.0, "shared default must be untouched"


async def test_bulk_member_update_over_max_batch_is_400(
    proxy_client, prisma, scratch, world
):
    """More than the 500-member cap of user_ids is rejected 400."""
    await create_scratch_team(prisma, scratch.prefix, organization_id=world.org_a_id)
    user_ids = [f"{scratch.prefix}-u{i}" for i in range(501)]
    resp = await proxy_client.patch(
        ROUTE.format(team_id=scratch.prefix),
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"user_ids": user_ids, "update_fields": {"role": "user"}},
    )
    assert resp.status_code == 400, resp.text
