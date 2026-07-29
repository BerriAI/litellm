import pytest

from .actors import Actor

pytestmark = pytest.mark.asyncio(loop_scope="session")


# POST /key/service-account/generate. PROXY_ADMIN always passes. ORG_ADMIN-role
# callers are stopped 401 by the management-route gate (the body carries a
# team_id but no organization_id, so the org-admin route branch never matches).
# INTERNAL_USER-role callers reach the handler: a team admin of the target team
# passes (200); a "user"-role member is 401 (no service-account-generate
# permission); a non-member is 400 ("not assigned to team"). A request with no
# team_id is 400 ("team_id is required") for every actor that reaches the handler.
_SCENARIOS = [
    ("own/proxy_admin", Actor.PROXY_ADMIN, "own", 200),
    ("own/org_admin", Actor.ORG_ADMIN, "own", 401),
    ("own/team_admin", Actor.TEAM_ADMIN, "own", 200),
    ("own/internal_user", Actor.INTERNAL_USER, "own", 401),
    ("own/owner", Actor.OWNER, "own", 401),
    ("own/unrelated_same_org", Actor.UNRELATED_SAME_ORG, "own", 401),
    ("own/cross_org_user", Actor.CROSS_ORG_USER, "own", 400),
    ("own/service_account", Actor.SERVICE_ACCOUNT, "own", 401),
    ("own/org_b_admin", Actor.ORG_B_ADMIN, "own", 401),
    ("cross_org/proxy_admin", Actor.PROXY_ADMIN, "cross_org", 200),
    ("cross_org/org_admin", Actor.ORG_ADMIN, "cross_org", 401),
    ("cross_org/team_admin", Actor.TEAM_ADMIN, "cross_org", 400),
    ("cross_org/internal_user", Actor.INTERNAL_USER, "cross_org", 400),
    ("cross_org/cross_org_user", Actor.CROSS_ORG_USER, "cross_org", 401),
    ("cross_org/org_b_admin", Actor.ORG_B_ADMIN, "cross_org", 401),
    ("none/proxy_admin", Actor.PROXY_ADMIN, "none", 400),
    ("none/org_admin", Actor.ORG_ADMIN, "none", 401),
    ("none/team_admin", Actor.TEAM_ADMIN, "none", 400),
    ("none/internal_user", Actor.INTERNAL_USER, "none", 400),
    ("none/cross_org_user", Actor.CROSS_ORG_USER, "none", 400),
]


@pytest.mark.parametrize(
    "actor,team_target,expected_status",
    [(a, t, s) for (_id, a, t, s) in _SCENARIOS],
    ids=[s[0] for s in _SCENARIOS],
)
async def test_key_service_account_generate_authz_matrix(
    actor: Actor,
    team_target: str,
    expected_status: int,
    proxy_client,
    prisma,
    scratch,
    world,
):
    caller = world.keys[actor]
    team_id = {
        "own": world.team_alpha_id,
        "cross_org": world.team_beta_id,
        "none": None,
    }[team_target]

    body = {"key_alias": scratch.prefix}
    if team_id is not None:
        body["team_id"] = team_id

    resp = await proxy_client.post(
        "/key/service-account/generate",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json=body,
    )
    assert (
        resp.status_code == expected_status
    ), f"{actor.value} {team_target}: {resp.status_code} {resp.text}"

    rows = await prisma.db.litellm_verificationtoken.find_many(
        where={"key_alias": scratch.prefix}
    )
    if expected_status == 200:
        assert len(rows) == 1
        # A service-account key belongs to the team, not a user.
        assert rows[0].user_id is None
        assert rows[0].team_id == team_id
    else:
        assert rows == [], f"{actor.value}: denied but key row leaked"


async def test_key_service_account_generate_unknown_team_is_400(
    proxy_client, prisma, scratch, world
):
    """A team_id absent from the database is rejected 400."""
    resp = await proxy_client.post(
        "/key/service-account/generate",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"key_alias": scratch.prefix, "team_id": scratch.tag("no-such-team")},
    )
    assert resp.status_code == 400, resp.text
    rows = await prisma.db.litellm_verificationtoken.find_many(
        where={"key_alias": scratch.prefix}
    )
    assert rows == []
