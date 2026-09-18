import uuid

import pytest

from .actors import Actor
from .conftest import create_scratch_team

pytestmark = pytest.mark.asyncio(loop_scope="session")

_SEED_SPEND = 5.0
_RESET_TO = 2.0


# POST /team/{team_id}/member/{user_id}/reset_spend. The handler gate is
# _verify_team_access (proxy admin / team admin of this team / org admin of
# the team's org) — the same gate /team/member_update uses, so this mirrors
# that file's matrix exactly.
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


async def _seed_target(prisma, world, shape: str, team_id: str, member_id: str) -> None:
    if shape == "alpha":
        await create_scratch_team(
            prisma,
            team_id,
            organization_id=world.org_a_id,
            admin_user_ids=[world.keys[Actor.TEAM_ADMIN].user_id],
        )
    elif shape == "beta":
        await create_scratch_team(prisma, team_id, organization_id=world.org_b_id)
    else:  # pragma: no cover - guard
        pytest.fail(f"unknown shape={shape}")
    await prisma.db.litellm_teammembership.create(
        data={"user_id": member_id, "team_id": team_id, "spend": _SEED_SPEND}
    )


@pytest.mark.parametrize(
    "actor,shape,expected_status",
    [(a, sh, s) for (_id, a, sh, s) in _MATRIX],
    ids=[s[0] for s in _MATRIX],
)
async def test_team_member_reset_spend_authz_matrix(
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

    resp = await proxy_client.post(
        f"/team/{scratch.prefix}/member/{member_id}/reset_spend",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={"reset_to": _RESET_TO},
    )
    assert (
        resp.status_code == expected_status
    ), f"{actor.value} {shape}: {resp.status_code} {resp.text}"

    row = await prisma.db.litellm_teammembership.find_unique(
        where={"user_id_team_id": {"user_id": member_id, "team_id": scratch.prefix}}
    )
    assert row is not None
    if expected_status == 200:
        assert row.spend == _RESET_TO
    else:
        assert row.spend == _SEED_SPEND, "denied but spend reset"


async def test_team_member_reset_spend_missing_team_is_404(proxy_client, world):
    resp = await proxy_client.post(
        f"/team/behavior-pin-no-such-team/member/{uuid.uuid4().hex}/reset_spend",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"reset_to": 0.0},
    )
    assert resp.status_code == 404, resp.text


async def test_team_member_reset_spend_missing_membership_is_404(
    proxy_client, prisma, scratch, world
):
    """A well-formed team but a user_id with no LiteLLM_TeamMembership row is 404."""
    await create_scratch_team(prisma, scratch.prefix, organization_id=world.org_a_id)
    resp = await proxy_client.post(
        f"/team/{scratch.prefix}/member/{uuid.uuid4().hex}/reset_spend",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"reset_to": 0.0},
    )
    assert resp.status_code == 404, resp.text


async def test_team_member_reset_spend_above_current_spend_is_400(
    proxy_client, prisma, scratch, world
):
    member_id = scratch.tag("member")
    await create_scratch_team(prisma, scratch.prefix, organization_id=world.org_a_id)
    await prisma.db.litellm_teammembership.create(
        data={"user_id": member_id, "team_id": scratch.prefix, "spend": 1.0}
    )
    resp = await proxy_client.post(
        f"/team/{scratch.prefix}/member/{member_id}/reset_spend",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"reset_to": 5.0},
    )
    assert resp.status_code == 400, resp.text


async def test_team_member_reset_spend_team_admin_cannot_reset_own_spend(
    proxy_client, prisma, scratch, world
):
    """A team admin targeting their own LiteLLM_TeamMembership row is 403: unchecked, an
    admin could repeatedly zero their own spend right before it crosses their per-member
    cap, consuming the shared team budget without the configured limit ever binding."""
    team_admin = world.keys[Actor.TEAM_ADMIN]
    await create_scratch_team(
        prisma,
        scratch.prefix,
        organization_id=world.org_a_id,
        admin_user_ids=[team_admin.user_id],
    )
    await prisma.db.litellm_teammembership.create(
        data={"user_id": team_admin.user_id, "team_id": scratch.prefix, "spend": _SEED_SPEND}
    )
    resp = await proxy_client.post(
        f"/team/{scratch.prefix}/member/{team_admin.user_id}/reset_spend",
        headers={"Authorization": f"Bearer {team_admin.cleartext}"},
        json={"reset_to": 0.0},
    )
    assert resp.status_code == 403, resp.text
    row = await prisma.db.litellm_teammembership.find_unique(
        where={"user_id_team_id": {"user_id": team_admin.user_id, "team_id": scratch.prefix}}
    )
    assert row is not None and row.spend == _SEED_SPEND, "denied but spend reset"
