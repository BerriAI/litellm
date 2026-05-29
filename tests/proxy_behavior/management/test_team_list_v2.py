from typing import FrozenSet, Optional

import pytest

from .actors import Actor

pytestmark = pytest.mark.asyncio(loop_scope="session")


def _seeded(team_ids: set, world) -> set:
    known = {
        world.team_alpha_id: "alpha",
        world.team_beta_id: "beta",
        world.team_gamma_id: "gamma",
    }
    return {known[t] for t in team_ids if t in known}


async def _v2_team_ids(proxy_client, caller_cleartext: str, extra: str = "") -> set:
    """Walk every /v2/team/list page and collect the returned team_ids."""
    ids: set = set()
    page = 1
    while True:
        resp = await proxy_client.get(
            f"/v2/team/list?page={page}&page_size=100{extra}",
            headers={"Authorization": f"Bearer {caller_cleartext}"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        teams = body.get("teams", []) or []
        for t in teams:
            tid = t.get("team_id") if isinstance(t, dict) else None
            if tid:
                ids.add(tid)
        if page * 100 >= (body.get("total") or 0) or not teams:
            return ids
        page += 1


# GET /v2/team/list is an info route reachable by every actor, but
# _enforce_list_team_v2_access still gates a BARE query: a proxy admin sees
# all teams, an org admin sees its orgs' teams, and a regular user — who has
# passed no user_id filter — is rejected 401 ("only admins can query all
# teams"). A regular user must scope the query to its own user_id.
_BARE = [
    ("proxy_admin", Actor.PROXY_ADMIN, 200, frozenset({"alpha", "beta", "gamma"})),
    ("org_admin", Actor.ORG_ADMIN, 200, frozenset({"alpha", "gamma"})),
    ("org_b_admin", Actor.ORG_B_ADMIN, 200, frozenset({"beta"})),
    ("team_admin", Actor.TEAM_ADMIN, 401, None),
    ("internal_user", Actor.INTERNAL_USER, 401, None),
    ("owner", Actor.OWNER, 401, None),
    ("unrelated_same_org", Actor.UNRELATED_SAME_ORG, 401, None),
    ("cross_org_user", Actor.CROSS_ORG_USER, 401, None),
    ("service_account", Actor.SERVICE_ACCOUNT, 401, None),
]


@pytest.mark.parametrize(
    "actor,expected_status,expected_visible",
    [(a, s, v) for (_id, a, s, v) in _BARE],
    ids=[s[0] for s in _BARE],
)
async def test_team_list_v2_bare(
    actor: Actor,
    expected_status: int,
    expected_visible: Optional[FrozenSet[str]],
    proxy_client,
    world,
):
    caller = world.keys[actor]
    if expected_status != 200:
        resp = await proxy_client.get(
            "/v2/team/list",
            headers={"Authorization": f"Bearer {caller.cleartext}"},
        )
        assert resp.status_code == expected_status, resp.text
        return

    visible = _seeded(await _v2_team_ids(proxy_client, caller.cleartext), world)
    assert visible == set(
        expected_visible
    ), f"{actor.value}: expected {sorted(expected_visible)}, got {sorted(visible)}"


# A regular user scoping the query to its own user_id is allowed, and sees
# exactly the teams it belongs to.
_OWN = {
    Actor.TEAM_ADMIN: frozenset({"alpha"}),
    Actor.INTERNAL_USER: frozenset({"alpha"}),
    Actor.OWNER: frozenset({"alpha"}),
    Actor.UNRELATED_SAME_ORG: frozenset({"alpha"}),
    Actor.CROSS_ORG_USER: frozenset({"beta"}),
    Actor.SERVICE_ACCOUNT: frozenset({"alpha"}),
}


@pytest.mark.parametrize(
    "actor,expected_visible", list(_OWN.items()), ids=[a.value for a in _OWN]
)
async def test_team_list_v2_own_user_id_query(
    actor: Actor, expected_visible: FrozenSet[str], proxy_client, world
):
    caller = world.keys[actor]
    visible = _seeded(
        await _v2_team_ids(
            proxy_client, caller.cleartext, f"&user_id={caller.user_id}"
        ),
        world,
    )
    assert visible == set(
        expected_visible
    ), f"{actor.value}: expected {sorted(expected_visible)}, got {sorted(visible)}"


async def test_team_list_v2_user_id_filter_other_user_is_401(proxy_client, world):
    """A regular user filtering by another user's user_id is rejected 401."""
    resp = await proxy_client.get(
        f"/v2/team/list?user_id={world.keys[Actor.OWNER].user_id}",
        headers={
            "Authorization": f"Bearer {world.keys[Actor.INTERNAL_USER].cleartext}"
        },
    )
    assert resp.status_code == 401, resp.text


async def test_team_list_v2_org_filter_foreign_org_is_403(proxy_client, world):
    """An org admin filtering by an organization it does not administer is 403."""
    resp = await proxy_client.get(
        f"/v2/team/list?organization_id={world.org_b_id}",
        headers={"Authorization": f"Bearer {world.keys[Actor.ORG_ADMIN].cleartext}"},
    )
    assert resp.status_code == 403, resp.text


async def test_team_list_v2_invalid_status_is_400(proxy_client, world):
    """status accepts only 'deleted' — any other value is 400."""
    resp = await proxy_client.get(
        "/v2/team/list?status=bogus",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
    )
    assert resp.status_code == 400, resp.text
