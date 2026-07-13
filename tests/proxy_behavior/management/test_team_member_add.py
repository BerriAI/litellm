import litellm
import pytest

from .actors import Actor
from .conftest import create_scratch_team

pytestmark = pytest.mark.asyncio(loop_scope="session")


# POST /team/member_add — actor x team-shape matrix, pinned against
# _validate_team_member_add_permissions: PROXY_ADMIN, the team's team admin,
# or an org admin of the team's org may add members; everyone else is 403.
# Unlike /team/update there is no route gate in front, so the team-admin
# branch is reachable (TEAM_ADMIN, an internal_user, is allowed on its team).
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


async def _seed_target(prisma, world, shape: str, team_id: str) -> None:
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


def _member_ids(row) -> list:
    return [m["user_id"] for m in (row.members_with_roles or [])]


@pytest.mark.parametrize(
    "actor,shape,expected_status",
    [(a, sh, s) for (_id, a, sh, s) in _MATRIX],
    ids=[s[0] for s in _MATRIX],
)
async def test_team_member_add_authz_matrix(
    actor: Actor,
    shape: str,
    expected_status: int,
    proxy_client,
    prisma,
    scratch,
    world,
):
    await _seed_target(prisma, world, shape, scratch.prefix)
    caller = world.keys[actor]
    new_member_id = scratch.tag("newmember")

    resp = await proxy_client.post(
        "/team/member_add",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={
            "team_id": scratch.prefix,
            "member": {"user_id": new_member_id, "role": "user"},
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
        assert new_member_id in _member_ids(row)
    else:
        assert new_member_id not in _member_ids(row), "denied but member added"


# Available-team self-join: a non-admin caller may add ITSELF to a team listed
# in litellm.default_internal_user_params["available_teams"], but the bypass
# must not escalate to role=admin or inject another user.
_SELF_JOIN = [
    ("self_as_user", "self", "user", 200),
    ("self_as_admin", "self", "admin", 403),
    ("other_as_user", "other", "user", 403),
]


@pytest.mark.parametrize(
    "who,role,expected_status",
    [(w, r, s) for (_id, w, r, s) in _SELF_JOIN],
    ids=[s[0] for s in _SELF_JOIN],
)
async def test_team_member_add_available_team_self_join(
    who: str,
    role: str,
    expected_status: int,
    proxy_client,
    prisma,
    scratch,
    world,
    monkeypatch,
):
    # Org-less team with no admins: the INTERNAL_USER caller is neither team
    # nor org admin, so it lands on the available-team branch.
    await create_scratch_team(prisma, scratch.prefix)
    monkeypatch.setattr(
        litellm, "default_internal_user_params", {"available_teams": [scratch.prefix]}
    )

    caller = world.keys[Actor.INTERNAL_USER]
    member_id = caller.user_id if who == "self" else world.keys[Actor.OWNER].user_id

    resp = await proxy_client.post(
        "/team/member_add",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={
            "team_id": scratch.prefix,
            "member": {"user_id": member_id, "role": role},
        },
    )
    assert (
        resp.status_code == expected_status
    ), f"{who}/{role}: {resp.status_code} {resp.text}"

    row = await prisma.db.litellm_teamtable.find_unique(
        where={"team_id": scratch.prefix}
    )
    assert row is not None
    if expected_status == 200:
        assert member_id in _member_ids(row)
    else:
        assert member_id not in _member_ids(row), "denied but member added"
