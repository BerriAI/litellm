import pytest

from litellm.proxy._types import LitellmUserRoles

from .actors import Actor
from .conftest import create_scratch_actor, create_scratch_team

pytestmark = pytest.mark.asyncio(loop_scope="session")


# POST /team/update — actor x team-shape matrix (shapes built by _seed_target).
# Each request carries the team's own organization_id so a non-proxy-admin can
# reach the org-scoped branch of the route-permission gate (401 on denial),
# which fronts the handler's _verify_team_access. Only PROXY_ADMIN and an
# ORG_ADMIN of the team's org pass: an internal_user team admin is filtered by
# the route gate before _verify_team_access's team-admin branch is reached.
MARKER_ALIAS = "behavior-pin-update-marker-alias"

_MATRIX = [
    ("alpha/proxy_admin", Actor.PROXY_ADMIN, "alpha", 200),
    ("alpha/org_admin", Actor.ORG_ADMIN, "alpha", 200),
    ("alpha/team_admin", Actor.TEAM_ADMIN, "alpha", 401),
    ("alpha/internal_user", Actor.INTERNAL_USER, "alpha", 401),
    ("alpha/owner", Actor.OWNER, "alpha", 401),
    ("alpha/unrelated_same_org", Actor.UNRELATED_SAME_ORG, "alpha", 401),
    ("alpha/cross_org_user", Actor.CROSS_ORG_USER, "alpha", 401),
    ("alpha/service_account", Actor.SERVICE_ACCOUNT, "alpha", 401),
    ("alpha/org_b_admin", Actor.ORG_B_ADMIN, "alpha", 401),
    ("beta/proxy_admin", Actor.PROXY_ADMIN, "beta", 200),
    ("beta/org_admin", Actor.ORG_ADMIN, "beta", 401),
    ("beta/team_admin", Actor.TEAM_ADMIN, "beta", 401),
    ("beta/internal_user", Actor.INTERNAL_USER, "beta", 401),
    ("beta/owner", Actor.OWNER, "beta", 401),
    ("beta/unrelated_same_org", Actor.UNRELATED_SAME_ORG, "beta", 401),
    ("beta/cross_org_user", Actor.CROSS_ORG_USER, "beta", 401),
    ("beta/service_account", Actor.SERVICE_ACCOUNT, "beta", 401),
    ("beta/org_b_admin", Actor.ORG_B_ADMIN, "beta", 200),
]


async def _seed_target(prisma, world, shape: str, team_id: str) -> str:
    """Raw-seed the scratch target team; returns its organization_id."""
    if shape == "alpha":
        await create_scratch_team(
            prisma,
            team_id,
            organization_id=world.org_a_id,
            admin_user_ids=[world.keys[Actor.TEAM_ADMIN].user_id],
            member_user_ids=[
                world.keys[Actor.INTERNAL_USER].user_id,
                world.keys[Actor.OWNER].user_id,
                world.keys[Actor.UNRELATED_SAME_ORG].user_id,
                world.keys[Actor.SERVICE_ACCOUNT].user_id,
            ],
        )
        return world.org_a_id
    if shape == "beta":
        await create_scratch_team(
            prisma,
            team_id,
            organization_id=world.org_b_id,
            member_user_ids=[world.keys[Actor.CROSS_ORG_USER].user_id],
        )
        return world.org_b_id
    pytest.fail(f"unknown shape={shape}")  # pragma: no cover


@pytest.mark.parametrize(
    "actor,shape,expected_status",
    [(a, sh, s) for (_id, a, sh, s) in _MATRIX],
    ids=[s[0] for s in _MATRIX],
)
async def test_team_update_authz_matrix(
    actor: Actor,
    shape: str,
    expected_status: int,
    proxy_client,
    prisma,
    scratch,
    world,
):
    org_id = await _seed_target(prisma, world, shape, scratch.prefix)
    caller = world.keys[actor]

    resp = await proxy_client.post(
        "/team/update",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={
            "team_id": scratch.prefix,
            "team_alias": MARKER_ALIAS,
            "organization_id": org_id,
        },
    )
    assert (
        resp.status_code == expected_status
    ), f"{actor.value} {shape}: {resp.status_code} {resp.text}"

    row = await prisma.db.litellm_teamtable.find_unique(
        where={"team_id": scratch.prefix}
    )
    assert row is not None
    if expected_status == 200:
        assert row.team_alias == MARKER_ALIAS
    else:
        assert row.team_alias != MARKER_ALIAS, "denied but team mutated"


async def test_team_update_requires_proxy_admin_without_org_context(
    proxy_client, prisma, scratch, world
):
    """With no organization_id in the body the route gate has no org context
    and falls back to proxy-admin-only: an org admin of the team's own org
    is 401, PROXY_ADMIN is 200."""
    await _seed_target(prisma, world, "alpha", scratch.prefix)

    denied = await proxy_client.post(
        "/team/update",
        headers={"Authorization": f"Bearer {world.keys[Actor.ORG_ADMIN].cleartext}"},
        json={"team_id": scratch.prefix, "team_alias": MARKER_ALIAS},
    )
    assert denied.status_code == 401, denied.text

    allowed = await proxy_client.post(
        "/team/update",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"team_id": scratch.prefix, "team_alias": MARKER_ALIAS},
    )
    assert allowed.status_code == 200, allowed.text


# Relocation gate — moving a team to a different org. The scratch team starts
# in ORG_A; each scenario relocates it to ORG_B. PROXY_ADMIN bypasses;
# ORG_B_ADMIN clears the route gate (dest-org admin) but fails
# _verify_team_access on the source team (403); the rest fail the route gate
# (401). The relocation-*allowed* branch (caller is org admin of both orgs) is
# covered by test_team_update_org_relocation_allowed_for_dual_org_admin below.
_RELOCATION = [
    ("proxy_admin", Actor.PROXY_ADMIN, 200),
    ("org_b_admin", Actor.ORG_B_ADMIN, 403),
    ("org_admin", Actor.ORG_ADMIN, 401),
    ("team_admin", Actor.TEAM_ADMIN, 401),
    ("internal_user", Actor.INTERNAL_USER, 401),
]


@pytest.mark.parametrize(
    "actor,expected_status",
    [(a, s) for (_id, a, s) in _RELOCATION],
    ids=[s[0] for s in _RELOCATION],
)
async def test_team_update_org_relocation_gate(
    actor: Actor,
    expected_status: int,
    proxy_client,
    prisma,
    scratch,
    world,
):
    await _seed_target(prisma, world, "alpha", scratch.prefix)
    caller = world.keys[actor]

    resp = await proxy_client.post(
        "/team/update",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={"team_id": scratch.prefix, "organization_id": world.org_b_id},
    )
    assert (
        resp.status_code == expected_status
    ), f"{actor.value}: {resp.status_code} {resp.text}"

    row = await prisma.db.litellm_teamtable.find_unique(
        where={"team_id": scratch.prefix}
    )
    assert row is not None
    if expected_status == 200:
        assert row.organization_id == world.org_b_id
    else:
        assert row.organization_id == world.org_a_id, "denied but team relocated"


# Phase 4 F6 — explicit pin on the `_verify_team_access` 403 detail string
# when an org_admin clears the destination route gate but fails the source
# team's org-membership check. The relocation matrix above covers the
# status; this guard turns a silent rename of the helper's exception detail
# into a CI red.
async def test_team_update_org_b_admin_relocation_rejection_detail(
    proxy_client, prisma, scratch, world
):
    await _seed_target(prisma, world, "alpha", scratch.prefix)
    resp = await proxy_client.post(
        "/team/update",
        headers={"Authorization": f"Bearer {world.keys[Actor.ORG_B_ADMIN].cleartext}"},
        json={"team_id": scratch.prefix, "organization_id": world.org_b_id},
    )
    assert resp.status_code == 403, resp.text
    assert "do not have access to this team" in resp.text, resp.text


async def test_team_update_org_relocation_allowed_for_dual_org_admin(
    proxy_client, prisma, scratch, world
):
    """Relocation-allowed branch: a caller who is org admin of BOTH the source
    and destination org may relocate a team between them. Completes the
    _RELOCATION matrix, whose allowed branch PR2 left open — no seeded actor is
    a dual-org admin, so one is minted with create_scratch_actor."""
    actor = await create_scratch_actor(
        prisma,
        scratch.prefix,
        user_role=LitellmUserRoles.ORG_ADMIN.value,
        org_admin_of=(world.org_a_id, world.org_b_id),
    )
    team_id = await create_scratch_team(
        prisma, scratch.tag("team"), organization_id=world.org_a_id
    )

    resp = await proxy_client.post(
        "/team/update",
        headers={"Authorization": f"Bearer {actor.cleartext}"},
        json={"team_id": team_id, "organization_id": world.org_b_id},
    )
    assert resp.status_code == 200, resp.text

    row = await prisma.db.litellm_teamtable.find_unique(where={"team_id": team_id})
    assert row is not None
    assert (
        row.organization_id == world.org_b_id
    ), "dual-org admin relocation not applied"
