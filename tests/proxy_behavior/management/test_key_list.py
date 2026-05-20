from typing import FrozenSet

import pytest

from .actors import Actor

pytestmark = pytest.mark.asyncio(loop_scope="session")


# Pinned default visibility for /key/list (no filter params): each actor's
# expected set of seeded actor keys.
_VISIBILITY = {
    Actor.PROXY_ADMIN: frozenset(Actor),
    Actor.ORG_ADMIN: frozenset({Actor.ORG_ADMIN}),
    Actor.TEAM_ADMIN: frozenset({Actor.TEAM_ADMIN}),
    Actor.INTERNAL_USER: frozenset({Actor.INTERNAL_USER}),
    Actor.OWNER: frozenset({Actor.OWNER}),
    Actor.UNRELATED_SAME_ORG: frozenset({Actor.UNRELATED_SAME_ORG}),
    Actor.CROSS_ORG_USER: frozenset({Actor.CROSS_ORG_USER}),
    Actor.SERVICE_ACCOUNT: frozenset({Actor.SERVICE_ACCOUNT}),
}


@pytest.mark.parametrize(
    "actor,expected_visible",
    list(_VISIBILITY.items()),
    ids=[a.value for a in _VISIBILITY],
)
async def test_key_list_visibility(
    actor: Actor, expected_visible: FrozenSet[Actor], proxy_client, world
):
    caller = world.keys[actor]
    hashed_to_actor = {world.keys[a].hashed: a for a in Actor}

    resp = await proxy_client.get(
        "/key/list?size=100",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
    )
    assert resp.status_code == 200, f"{actor.value}: {resp.text}"

    returned_hashes = {
        (entry.get("token") if isinstance(entry, dict) else entry)
        for entry in resp.json().get("keys", [])
    }
    visible_seeded = {
        hashed_to_actor[h] for h in returned_hashes if h in hashed_to_actor
    }
    assert visible_seeded == set(expected_visible), (
        f"{actor.value}: expected {sorted(a.value for a in expected_visible)}, "
        f"got {sorted(a.value for a in visible_seeded)}"
    )
