import uuid

import pytest

from .actors import Actor
from .conftest import create_scratch_team

pytestmark = pytest.mark.asyncio(loop_scope="session")

_SEED_SPEND = 5.0
_TEAM_DEFAULT_MAX_BUDGET = 100.0
_CUSTOM_MAX_BUDGET = 50.0

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
    ("beta/org_b_admin", Actor.ORG_B_ADMIN, "beta", 200),
]


async def _seed_budget(prisma, budget_id: str, max_budget: float) -> str:
    await prisma.db.litellm_budgettable.create(
        data={
            "budget_id": budget_id,
            "max_budget": max_budget,
            "created_by": "phase4-scratch",
            "updated_by": "phase4-scratch",
        }
    )
    return budget_id


async def _seed_team_with_default_budget(prisma, world, shape: str, team_id: str, scratch) -> str:
    default_budget_id = await _seed_budget(prisma, scratch.tag("team-default-budget"), _TEAM_DEFAULT_MAX_BUDGET)
    metadata = {"team_member_budget_id": default_budget_id}
    if shape == "alpha":
        await create_scratch_team(
            prisma,
            team_id,
            organization_id=world.org_a_id,
            admin_user_ids=[world.keys[Actor.TEAM_ADMIN].user_id],
            metadata=metadata,
        )
    elif shape == "beta":
        await create_scratch_team(prisma, team_id, organization_id=world.org_b_id, metadata=metadata)
    else:  # pragma: no cover - guard
        pytest.fail(f"unknown shape={shape}")
    return default_budget_id


async def _seed_custom_member(prisma, team_id: str, member_id: str, scratch) -> str:
    custom_budget_id = await _seed_budget(prisma, scratch.tag("custom-budget"), _CUSTOM_MAX_BUDGET)
    await prisma.db.litellm_teammembership.create(
        data={
            "user_id": member_id,
            "team_id": team_id,
            "spend": _SEED_SPEND,
            "litellm_budget_table": {"connect": {"budget_id": custom_budget_id}},
        }
    )
    return custom_budget_id


async def _membership(prisma, team_id: str, member_id: str):
    row = await prisma.db.litellm_teammembership.find_unique(
        where={"user_id_team_id": {"user_id": member_id, "team_id": team_id}}
    )
    assert row is not None
    return row


@pytest.mark.parametrize(
    "actor,shape,expected_status",
    [(a, sh, s) for (_id, a, sh, s) in _MATRIX],
    ids=[s[0] for s in _MATRIX],
)
async def test_team_member_reset_budget_authz_matrix(
    actor: Actor,
    shape: str,
    expected_status: int,
    proxy_client,
    prisma,
    scratch,
    world,
):
    member_id = scratch.tag("member")
    default_budget_id = await _seed_team_with_default_budget(prisma, world, shape, scratch.prefix, scratch)
    custom_budget_id = await _seed_custom_member(prisma, scratch.prefix, member_id, scratch)
    caller = world.keys[actor]

    resp = await proxy_client.post(
        f"/team/{scratch.prefix}/member/{member_id}/reset_budget",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
    )
    assert resp.status_code == expected_status, f"{actor.value} {shape}: {resp.status_code} {resp.text}"

    row = await _membership(prisma, scratch.prefix, member_id)
    assert row.spend == _SEED_SPEND, "reset_budget must never touch spend"
    if expected_status == 200:
        assert row.budget_id == default_budget_id
        body = resp.json()
        assert body["budget_id"] == default_budget_id
        assert body["previous_budget_id"] == custom_budget_id
        assert body["budget_source"] == "team_default"
    else:
        assert row.budget_id == custom_budget_id, "denied but budget relinked"


async def test_team_member_reset_budget_leaves_shared_default_row_untouched(proxy_client, prisma, scratch, world):
    member_id = scratch.tag("member")
    default_budget_id = await _seed_team_with_default_budget(prisma, world, "alpha", scratch.prefix, scratch)
    await _seed_custom_member(prisma, scratch.prefix, member_id, scratch)

    resp = await proxy_client.post(
        f"/team/{scratch.prefix}/member/{member_id}/reset_budget",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
    )
    assert resp.status_code == 200, resp.text

    default_row = await prisma.db.litellm_budgettable.find_unique(where={"budget_id": default_budget_id})
    assert default_row is not None and default_row.max_budget == _TEAM_DEFAULT_MAX_BUDGET

    info = await proxy_client.get(
        f"/team/info?team_id={scratch.prefix}",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
    )
    assert info.status_code == 200, info.text
    memberships = {tm["user_id"]: tm for tm in info.json()["team_memberships"]}
    assert memberships[member_id]["budget_source"] == "team_default"
    assert memberships[member_id]["litellm_budget_table"]["max_budget"] == _TEAM_DEFAULT_MAX_BUDGET


async def test_team_member_reset_budget_without_team_default_detaches_member(proxy_client, prisma, scratch, world):
    member_id = scratch.tag("member")
    await create_scratch_team(prisma, scratch.prefix, organization_id=world.org_a_id)
    await _seed_custom_member(prisma, scratch.prefix, member_id, scratch)

    resp = await proxy_client.post(
        f"/team/{scratch.prefix}/member/{member_id}/reset_budget",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["budget_id"] is None
    assert resp.json()["budget_source"] == "none"

    row = await _membership(prisma, scratch.prefix, member_id)
    assert row.budget_id is None
    assert row.spend == _SEED_SPEND


async def test_team_member_reset_budget_with_deleted_team_default_detaches_member(proxy_client, prisma, scratch, world):
    member_id = scratch.tag("member")
    await create_scratch_team(
        prisma,
        scratch.prefix,
        organization_id=world.org_a_id,
        metadata={"team_member_budget_id": scratch.tag("deleted-budget")},
    )
    await _seed_custom_member(prisma, scratch.prefix, member_id, scratch)

    resp = await proxy_client.post(
        f"/team/{scratch.prefix}/member/{member_id}/reset_budget",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["budget_id"] is None
    assert resp.json()["budget_source"] == "none"

    row = await _membership(prisma, scratch.prefix, member_id)
    assert row.budget_id is None


async def test_team_member_reset_budget_missing_team_is_404(proxy_client, world):
    resp = await proxy_client.post(
        f"/team/behavior-pin-no-such-team/member/{uuid.uuid4().hex}/reset_budget",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
    )
    assert resp.status_code == 404, resp.text


async def test_team_member_reset_budget_missing_membership_is_404(proxy_client, prisma, scratch, world):
    await create_scratch_team(prisma, scratch.prefix, organization_id=world.org_a_id)
    resp = await proxy_client.post(
        f"/team/{scratch.prefix}/member/{uuid.uuid4().hex}/reset_budget",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
    )
    assert resp.status_code == 404, resp.text
