import pytest

from .actors import Actor

pytestmark = pytest.mark.asyncio(loop_scope="session")


# GET /team/info — actor x team-target authz matrix, pinned against
# validate_membership(): a team is readable by a proxy admin, a key whose
# own team_id matches, a listed member, or an org admin of the team's org;
# everything else is 403. TEAM_GAMMA has no members, so only PROXY_ADMIN
# and ORG_A's org admin can read it.
_SCENARIOS = [
    ("alpha/proxy_admin", Actor.PROXY_ADMIN, "alpha", 200),
    ("alpha/org_admin", Actor.ORG_ADMIN, "alpha", 200),
    ("alpha/team_admin", Actor.TEAM_ADMIN, "alpha", 200),
    ("alpha/internal_user", Actor.INTERNAL_USER, "alpha", 200),
    ("alpha/owner", Actor.OWNER, "alpha", 200),
    ("alpha/unrelated_same_org", Actor.UNRELATED_SAME_ORG, "alpha", 200),
    ("alpha/cross_org_user", Actor.CROSS_ORG_USER, "alpha", 403),
    ("alpha/service_account", Actor.SERVICE_ACCOUNT, "alpha", 200),
    ("alpha/org_b_admin", Actor.ORG_B_ADMIN, "alpha", 403),
    ("gamma/proxy_admin", Actor.PROXY_ADMIN, "gamma", 200),
    ("gamma/org_admin", Actor.ORG_ADMIN, "gamma", 200),
    ("gamma/team_admin", Actor.TEAM_ADMIN, "gamma", 403),
    ("gamma/internal_user", Actor.INTERNAL_USER, "gamma", 403),
    ("gamma/owner", Actor.OWNER, "gamma", 403),
    ("gamma/unrelated_same_org", Actor.UNRELATED_SAME_ORG, "gamma", 403),
    ("gamma/cross_org_user", Actor.CROSS_ORG_USER, "gamma", 403),
    ("gamma/service_account", Actor.SERVICE_ACCOUNT, "gamma", 403),
    ("gamma/org_b_admin", Actor.ORG_B_ADMIN, "gamma", 403),
    ("beta/proxy_admin", Actor.PROXY_ADMIN, "beta", 200),
    ("beta/org_admin", Actor.ORG_ADMIN, "beta", 403),
    ("beta/team_admin", Actor.TEAM_ADMIN, "beta", 403),
    ("beta/internal_user", Actor.INTERNAL_USER, "beta", 403),
    ("beta/owner", Actor.OWNER, "beta", 403),
    ("beta/unrelated_same_org", Actor.UNRELATED_SAME_ORG, "beta", 403),
    ("beta/cross_org_user", Actor.CROSS_ORG_USER, "beta", 200),
    ("beta/service_account", Actor.SERVICE_ACCOUNT, "beta", 403),
    ("beta/org_b_admin", Actor.ORG_B_ADMIN, "beta", 200),
]


@pytest.mark.parametrize(
    "actor,target,expected_status",
    [(a, t, s) for (_id, a, t, s) in _SCENARIOS],
    ids=[s[0] for s in _SCENARIOS],
)
async def test_team_info_authz_matrix(
    actor: Actor, target: str, expected_status: int, proxy_client, world
):
    caller = world.keys[actor]
    target_team_id = {
        "alpha": world.team_alpha_id,
        "gamma": world.team_gamma_id,
        "beta": world.team_beta_id,
    }[target]

    resp = await proxy_client.get(
        f"/team/info?team_id={target_team_id}",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
    )
    assert (
        resp.status_code == expected_status
    ), f"{actor.value} -> {target}: {resp.status_code} {resp.text}"

    if expected_status == 200:
        body = resp.json()
        assert body["team_id"] == target_team_id
        assert body["team_info"]["team_id"] == target_team_id
