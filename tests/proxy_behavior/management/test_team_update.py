import pytest

from .actors import Actor
from .conftest import create_scratch_team

pytestmark = pytest.mark.asyncio(loop_scope="session")


# The target is a raw-seeded scratch team. Two fixed shapes:
#   alpha = ORG_A team; TEAM_ADMIN is its team admin, the other ORG_A
#           internal users are plain members.
#   beta  = ORG_B team; CROSS_ORG_USER is a plain member.
#
# /team/update is fronted by an org-contextual route-permission gate (401 on
# denial) BEFORE the handler's own _verify_team_access (403). Every request
# below carries organization_id = the team's own org, which is what lets a
# non-proxy-admin reach the gate's org-scoped branch.
#
# Pinned behavior — SURFACED, NOT ENDORSED:
#   - PROXY_ADMIN always passes.
#   - The route gate admits only an ORG_ADMIN of the org named in the body;
#     an internal_user is 401 regardless of team-admin status. So a team
#     admin canNOT update its own team via /team/update — _verify_team_
#     access's "team admin" branch is unreachable here because the route
#     gate filters every internal_user first.
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


# Relocation gate: moving a team to a *different* org. The scratch team starts
# in ORG_A; every scenario attempts to relocate it to ORG_B.
#   - PROXY_ADMIN: bypasses every gate -> 200.
#   - ORG_B_ADMIN: org admin of the destination, so it clears the route
#     gate, but it is not an admin of the source team -> 403 at
#     _verify_team_access.
#   - ORG_ADMIN / TEAM_ADMIN / INTERNAL_USER: not org admin of the
#     destination ORG_B -> 401 at the route gate.
# (Gap: the relocation-ALLOWED branch for a non-proxy-admin needs a caller
# who is org admin of both source and destination; no seeded actor is, so
# that branch is left to a later slice — see README.)
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
