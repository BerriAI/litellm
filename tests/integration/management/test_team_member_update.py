import pytest

from tests.integration._support.actors import Actor
from tests.integration._support.actors import create_scratch_team

pytestmark = pytest.mark.asyncio(loop_scope="session")


# POST /team/member_update — actor x team-shape matrix. The resources team is
# raw-seeded with a "user"-role member; each scenario tries to promote it to
# "admin". PROXY_ADMIN, the team's team admin, or an org admin of the team's
# org may update members; else 403. (The harness forces premium_user, so the
# promotion does not hit the admin-role premium gate.)
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


async def _seed_target(db, actors, shape: str, team_id: str, member_id: str) -> None:
    if shape == "alpha":
        await create_scratch_team(
            db,
            team_id,
            organization_id=actors.org_a_id,
            admin_user_ids=[actors.keys[Actor.TEAM_ADMIN].user_id],
            member_user_ids=[member_id],
        )
    elif shape == "beta":
        await create_scratch_team(
            db,
            team_id,
            organization_id=actors.org_b_id,
            member_user_ids=[member_id],
        )
    else:  # pragma: no cover - guard
        pytest.fail(f"unknown shape={shape}")


def _role_of(row, user_id: str):
    for m in row.members_with_roles or []:
        if m["user_id"] == user_id:
            return m["role"]
    return None


@pytest.mark.parametrize(
    "actor,shape,expected_status",
    [(a, sh, s) for (_id, a, sh, s) in _MATRIX],
    ids=[s[0] for s in _MATRIX],
)
async def test_team_member_update_authz_matrix(
    actor: Actor,
    shape: str,
    expected_status: int,
    gateway_client,
    db,
    resources,
    actors,
):
    member_id = resources.tag("member")
    await _seed_target(db, actors, shape, resources.prefix, member_id)
    caller = actors.keys[actor]

    resp = await gateway_client.post(
        "/team/member_update",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={"team_id": resources.prefix, "user_id": member_id, "role": "admin"},
    )
    assert (
        resp.status_code == expected_status
    ), f"{actor.value} {shape}: {resp.status_code} {resp.text}"

    row = await db.db.litellm_teamtable.find_unique(
        where={"team_id": resources.prefix}
    )
    assert row is not None
    if expected_status == 200:
        assert _role_of(row, member_id) == "admin"
    else:
        assert _role_of(row, member_id) == "user", "denied but role changed"
