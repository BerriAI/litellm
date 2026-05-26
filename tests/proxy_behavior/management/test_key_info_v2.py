import uuid

import pytest

from .actors import Actor

pytestmark = pytest.mark.asyncio(loop_scope="session")


# POST /v2/key/info resolves the posted keys, then drops any key the caller
# cannot see via _can_user_query_key_info — silently, no 403. A non-admin sees
# a key it owns (user_id match) or a key whose team it belongs to. The world's
# TEAM_ALPHA members all see each other's keys; CROSS_ORG_USER and the org
# admins see only their own. The request is posted with every world key, and
# the returned info set is asserted to equal the visible subset.
_ALPHA_KEYS = frozenset(
    {
        Actor.TEAM_ADMIN,
        Actor.INTERNAL_USER,
        Actor.OWNER,
        Actor.UNRELATED_SAME_ORG,
        Actor.SERVICE_ACCOUNT,
    }
)
_VISIBILITY = {
    Actor.PROXY_ADMIN: frozenset(Actor),
    Actor.ORG_ADMIN: frozenset({Actor.ORG_ADMIN}),
    Actor.TEAM_ADMIN: _ALPHA_KEYS,
    Actor.INTERNAL_USER: _ALPHA_KEYS,
    Actor.OWNER: _ALPHA_KEYS,
    Actor.UNRELATED_SAME_ORG: _ALPHA_KEYS,
    Actor.SERVICE_ACCOUNT: _ALPHA_KEYS,
    Actor.CROSS_ORG_USER: frozenset({Actor.CROSS_ORG_USER}),
    Actor.ORG_B_ADMIN: frozenset({Actor.ORG_B_ADMIN}),
}


@pytest.mark.parametrize(
    "actor,expected_visible",
    list(_VISIBILITY.items()),
    ids=[a.value for a in _VISIBILITY],
)
async def test_key_info_v2_visibility(actor, expected_visible, proxy_client, world):
    caller = world.keys[actor]
    user_id_to_actor = {world.keys[a].user_id: a for a in Actor}

    resp = await proxy_client.post(
        "/v2/key/info",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={"keys": [world.keys[a].cleartext for a in Actor]},
    )
    assert resp.status_code == 200, f"{actor.value}: {resp.status_code} {resp.text}"

    visible = {
        user_id_to_actor[entry["user_id"]]
        for entry in resp.json()["info"]
        if entry.get("user_id") in user_id_to_actor
    }
    assert visible == set(expected_visible), (
        f"{actor.value}: expected {sorted(a.value for a in expected_visible)}, "
        f"got {sorted(a.value for a in visible)}"
    )


async def test_key_info_v2_no_body_is_422(proxy_client, world):
    """A request with no body is a 422 — the handler has no keys to resolve."""
    resp = await proxy_client.post(
        "/v2/key/info",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
    )
    assert resp.status_code == 422, resp.text


async def test_key_info_v2_unknown_key_returns_empty_info(proxy_client, world):
    """Keys that resolve to no rows yield an empty info list, not an error."""
    resp = await proxy_client.post(
        "/v2/key/info",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"keys": ["sk-" + uuid.uuid4().hex]},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["info"] == []
