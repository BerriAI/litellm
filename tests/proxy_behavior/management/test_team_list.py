from typing import FrozenSet, Optional

import pytest

from .actors import Actor

pytestmark = pytest.mark.asyncio(loop_scope="session")


# The behavior DB may hold teams beyond the three seeded ones, so every
# assertion intersects the returned team_ids with the known seeded set.
def _seeded_visible(resp_json, world) -> set:
    known = {
        world.team_alpha_id: "alpha",
        world.team_beta_id: "beta",
        world.team_gamma_id: "gamma",
    }
    return {
        known[entry["team_id"]]
        for entry in resp_json
        if isinstance(entry, dict) and entry.get("team_id") in known
    }


# Family 1 — bare GET /team/list (no query params). _authorize_and_filter_teams
# gives a proxy admin every team, an org admin every team in its org(s), and
# everyone else the teams they belong to (LIT-5384: a bare query from a
# regular user used to be 401).
_BARE = [
    ("proxy_admin", Actor.PROXY_ADMIN, 200, {"alpha", "beta", "gamma"}),
    ("org_admin", Actor.ORG_ADMIN, 200, {"alpha", "gamma"}),
    ("team_admin", Actor.TEAM_ADMIN, 200, {"alpha"}),
    ("internal_user", Actor.INTERNAL_USER, 200, {"alpha"}),
    ("owner", Actor.OWNER, 200, {"alpha"}),
    ("unrelated_same_org", Actor.UNRELATED_SAME_ORG, 200, {"alpha"}),
    ("cross_org_user", Actor.CROSS_ORG_USER, 200, {"beta"}),
    ("service_account", Actor.SERVICE_ACCOUNT, 200, {"alpha"}),
    ("org_b_admin", Actor.ORG_B_ADMIN, 200, {"beta"}),
]


@pytest.mark.parametrize(
    "actor,expected_status,expected_visible",
    [(a, s, v) for (_id, a, s, v) in _BARE],
    ids=[s[0] for s in _BARE],
)
async def test_team_list_bare_authz(
    actor: Actor,
    expected_status: int,
    expected_visible: Optional[set],
    proxy_client,
    world,
):
    caller = world.keys[actor]
    resp = await proxy_client.get(
        "/team/list",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
    )
    assert (
        resp.status_code == expected_status
    ), f"{actor.value}: {resp.status_code} {resp.text}"

    if expected_status == 200:
        visible = _seeded_visible(resp.json(), world)
        assert visible == expected_visible, (
            f"{actor.value}: expected {sorted(expected_visible)}, "
            f"got {sorted(visible)}"
        )


# Family 2 — GET /team/list?user_id=<caller's own id> ("own query"). Every
# actor may query its own teams (200); the result is exactly the teams it
# belongs to. A user_id filter scopes proxy/org admins to their own
# membership too — the broad admin view from family 1 does not carry over.
_OWN = {
    Actor.PROXY_ADMIN: frozenset(),
    Actor.ORG_ADMIN: frozenset(),
    Actor.TEAM_ADMIN: frozenset({"alpha"}),
    Actor.INTERNAL_USER: frozenset({"alpha"}),
    Actor.OWNER: frozenset({"alpha"}),
    Actor.UNRELATED_SAME_ORG: frozenset({"alpha"}),
    Actor.CROSS_ORG_USER: frozenset({"beta"}),
    Actor.SERVICE_ACCOUNT: frozenset({"alpha"}),
    Actor.ORG_B_ADMIN: frozenset(),
}


@pytest.mark.parametrize(
    "actor,expected_visible",
    list(_OWN.items()),
    ids=[a.value for a in _OWN],
)
async def test_team_list_own_query(
    actor: Actor, expected_visible: FrozenSet[str], proxy_client, world
):
    caller = world.keys[actor]
    resp = await proxy_client.get(
        f"/team/list?user_id={caller.user_id}",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
    )
    assert resp.status_code == 200, f"{actor.value}: {resp.status_code} {resp.text}"

    visible = _seeded_visible(resp.json(), world)
    assert visible == set(expected_visible), (
        f"{actor.value}: expected {sorted(expected_visible)}, " f"got {sorted(visible)}"
    )


async def test_team_list_other_user_id_query_is_401(proxy_client, world):
    """A regular user filtering by another user's user_id is still rejected 401."""
    resp = await proxy_client.get(
        f"/team/list?user_id={world.keys[Actor.CROSS_ORG_USER].user_id}",
        headers={
            "Authorization": f"Bearer {world.keys[Actor.INTERNAL_USER].cleartext}"
        },
    )
    assert resp.status_code == 401, resp.text
