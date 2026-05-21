import pytest

from .actors import Actor

pytestmark = pytest.mark.asyncio(loop_scope="session")


# (id, actor, target, expected_status) for GET /team/info?team_id=...
# Targets are the three seeded teams:
#   alpha = TEAM_ALPHA (ORG_A, caller's seeded team for the ORG_A actors)
#   gamma = TEAM_GAMMA (ORG_A, a team with no actor members)
#   beta  = TEAM_BETA  (ORG_B, the cross-org team)
#
# Pinned against validate_membership() — read access is granted to:
#   - proxy admin (unscoped), OR
#   - a key whose own team_id == the requested team, OR
#   - a user_id listed in the team's members_with_roles, OR
#   - an org admin of the team's organization.
# Everything else is 403. team_info() re-raises the gate's HTTPException as
# a ProxyException preserving the status code.
#
# Notable pinned behaviors (surfaced, not endorsed):
#   - ORG_ADMIN reads any team in its own org (TEAM_ALPHA, TEAM_GAMMA) even
#     though it is a member of neither — org-admin is an org-wide read grant.
#   - TEAM_GAMMA has no members, so only PROXY_ADMIN and ORG_A's org admin
#     can see it; every TEAM_ALPHA member is denied.
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
