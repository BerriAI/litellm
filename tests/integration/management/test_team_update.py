import pytest

from litellm.proxy._types import LitellmUserRoles

from tests.integration._support.actors import Actor, SeededKey
from tests.integration._support.actors import create_scratch_actor, create_scratch_team

pytestmark = pytest.mark.asyncio(loop_scope="session")


# POST /team/update — actor x team-shape matrix (shapes built by _seed_target).
# The route is self-managed (LIT-5722), so every authenticated caller reaches
# update_team and denials are the handler's 403, never the route gate's 401.
# Only PROXY_ADMIN and an ORG_ADMIN of the team's org pass: a team admin is
# admitted by TeamAccess.strongest_role but then refused because no team field is
# enabled for team admins (team_admin_editable_team_fields defaults to empty).
MARKER_ALIAS = "behavior-pin-update-marker-alias"

_MATRIX = [
    ("alpha/proxy_admin", Actor.PROXY_ADMIN, "alpha", 200),
    ("alpha/org_admin", Actor.ORG_ADMIN, "alpha", 200),
    ("alpha/team_admin", Actor.TEAM_ADMIN, "alpha", 403),
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


_TEAM_MEMBER_ACTORS = (
    Actor.TEAM_ADMIN,
    Actor.INTERNAL_USER,
    Actor.OWNER,
    Actor.UNRELATED_SAME_ORG,
    Actor.SERVICE_ACCOUNT,
    Actor.CROSS_ORG_USER,
)


async def _callers(db, actors, prefix: str) -> dict[Actor, SeededKey]:
    team_members = {
        actor: await create_scratch_actor(
            db, prefix, user_role=LitellmUserRoles.INTERNAL_USER.value, suffix=actor.value
        )
        for actor in _TEAM_MEMBER_ACTORS
    }
    return {**actors.keys, **team_members}


async def _seed_target(db, actors, callers, shape: str, team_id: str) -> str:
    """Raw-seed the resources target team; returns its organization_id."""
    if shape == "alpha":
        await create_scratch_team(
            db,
            team_id,
            organization_id=actors.org_a_id,
            admin_user_ids=[callers[Actor.TEAM_ADMIN].user_id],
            member_user_ids=[
                callers[Actor.INTERNAL_USER].user_id,
                callers[Actor.OWNER].user_id,
                callers[Actor.UNRELATED_SAME_ORG].user_id,
                callers[Actor.SERVICE_ACCOUNT].user_id,
            ],
        )
        return actors.org_a_id
    if shape == "beta":
        await create_scratch_team(
            db,
            team_id,
            organization_id=actors.org_b_id,
            member_user_ids=[callers[Actor.CROSS_ORG_USER].user_id],
        )
        return actors.org_b_id
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
    gateway_client,
    db,
    resources,
    actors,
):
    callers = await _callers(db, actors, resources.prefix)
    org_id = await _seed_target(db, actors, callers, shape, resources.prefix)
    caller = callers[actor]

    resp = await gateway_client.post(
        "/team/update",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={
            "team_id": resources.prefix,
            "team_alias": MARKER_ALIAS,
            "organization_id": org_id,
        },
    )
    assert (
        resp.status_code == expected_status
    ), f"{actor.value} {shape}: {resp.status_code} {resp.text}"

    row = await db.db.litellm_teamtable.find_unique(
        where={"team_id": resources.prefix}
    )
    assert row is not None
    if expected_status == 200:
        assert row.team_alias == MARKER_ALIAS
    else:
        assert row.team_alias != MARKER_ALIAS, "denied but team mutated"


async def test_team_update_org_admin_resolved_from_team_without_org_context(
    gateway_client, db, resources, actors
):
    """With no organization_id in the body the route gate resolves the target
    team's org from team_id, so an org admin of the team's own org is allowed
    (200), same as PROXY_ADMIN. A team admin of that same team reaches the
    handler but is refused (403) until a proxy admin enables fields for team
    admins, and the response says so."""
    callers = await _callers(db, actors, resources.prefix)
    await _seed_target(db, actors, callers, "alpha", resources.prefix)

    allowed_org_admin = await gateway_client.post(
        "/team/update",
        headers={"Authorization": f"Bearer {actors.keys[Actor.ORG_ADMIN].cleartext}"},
        json={"team_id": resources.prefix, "team_alias": MARKER_ALIAS},
    )
    assert allowed_org_admin.status_code == 200, allowed_org_admin.text

    allowed_proxy_admin = await gateway_client.post(
        "/team/update",
        headers={"Authorization": f"Bearer {actors.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"team_id": resources.prefix, "team_alias": MARKER_ALIAS},
    )
    assert allowed_proxy_admin.status_code == 200, allowed_proxy_admin.text

    denied_team_admin = await gateway_client.post(
        "/team/update",
        headers={"Authorization": f"Bearer {callers[Actor.TEAM_ADMIN].cleartext}"},
        json={"team_id": resources.prefix, "team_alias": MARKER_ALIAS},
    )
    assert denied_team_admin.status_code == 403, denied_team_admin.text
    assert "cannot edit team settings" in denied_team_admin.text, denied_team_admin.text
    assert "Team admin editable fields" in denied_team_admin.text, denied_team_admin.text


# Relocation gate — moving a team to a different org. The resources team starts
# in actors.org_a_id; each scenario relocates it to actors.org_b_id. PROXY_ADMIN bypasses;
# ORG_B_ADMIN reaches the handler but holds no role on the source team (403);
# ORG_ADMIN holds the source team but not the destination org (403 from the
# relocation gate); the team admin is refused by the empty field allow-list and
# the internal user holds no role at all (403). The relocation-*allowed* branch
# (caller is org admin of both orgs) is covered by
# test_team_update_org_relocation_allowed_for_dual_org_admin below.
_RELOCATION = [
    ("proxy_admin", Actor.PROXY_ADMIN, 200),
    ("org_b_admin", Actor.ORG_B_ADMIN, 403),
    ("org_admin", Actor.ORG_ADMIN, 403),
    ("team_admin", Actor.TEAM_ADMIN, 403),
    ("internal_user", Actor.INTERNAL_USER, 403),
]


@pytest.mark.parametrize(
    "actor,expected_status",
    [(a, s) for (_id, a, s) in _RELOCATION],
    ids=[s[0] for s in _RELOCATION],
)
async def test_team_update_org_relocation_gate(
    actor: Actor,
    expected_status: int,
    gateway_client,
    db,
    resources,
    actors,
):
    callers = await _callers(db, actors, resources.prefix)
    await _seed_target(db, actors, callers, "alpha", resources.prefix)
    caller = callers[actor]

    resp = await gateway_client.post(
        "/team/update",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={"team_id": resources.prefix, "organization_id": actors.org_b_id},
    )
    assert (
        resp.status_code == expected_status
    ), f"{actor.value}: {resp.status_code} {resp.text}"

    row = await db.db.litellm_teamtable.find_unique(
        where={"team_id": resources.prefix}
    )
    assert row is not None
    if expected_status == 200:
        assert row.organization_id == actors.org_b_id
    else:
        assert row.organization_id == actors.org_a_id, "denied but team relocated"


# Phase 4 F6 — explicit pin on the `team_access_denied` 403 detail string
# when an org_admin clears the destination route gate but fails the source
# team's org-membership check. The relocation matrix above covers the
# status; this guard turns a silent rename of the helper's exception detail
# into a CI red.
async def test_team_update_org_b_admin_relocation_rejection_detail(
    gateway_client, db, resources, actors
):
    await _seed_target(db, actors, await _callers(db, actors, resources.prefix), "alpha", resources.prefix)
    resp = await gateway_client.post(
        "/team/update",
        headers={"Authorization": f"Bearer {actors.keys[Actor.ORG_B_ADMIN].cleartext}"},
        json={"team_id": resources.prefix, "organization_id": actors.org_b_id},
    )
    assert resp.status_code == 403, resp.text
    assert "do not have access to this team" in resp.text, resp.text


async def test_team_update_org_relocation_allowed_for_dual_org_admin(
    gateway_client, db, resources, actors
):
    """Relocation-allowed branch: a caller who is org admin of BOTH the source
    and destination org may relocate a team between them. Completes the
    _RELOCATION matrix, whose allowed branch PR2 left open — no seeded actor is
    a dual-org admin, so one is minted with create_scratch_actor."""
    actor = await create_scratch_actor(
        db,
        resources.prefix,
        user_role=LitellmUserRoles.ORG_ADMIN.value,
        org_admin_of=(actors.org_a_id, actors.org_b_id),
    )
    team_id = await create_scratch_team(
        db, resources.tag("team"), organization_id=actors.org_a_id
    )

    resp = await gateway_client.post(
        "/team/update",
        headers={"Authorization": f"Bearer {actor.cleartext}"},
        json={"team_id": team_id, "organization_id": actors.org_b_id},
    )
    assert resp.status_code == 200, resp.text

    row = await db.db.litellm_teamtable.find_unique(where={"team_id": team_id})
    assert row is not None
    assert (
        row.organization_id == actors.org_b_id
    ), "dual-org admin relocation not applied"
