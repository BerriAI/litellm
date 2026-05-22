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


async def _all_visible_hashes(proxy_client, caller_cleartext) -> set:
    """Walk every /key/list page — size is capped at 100 by the endpoint, so a
    single request can truncate PROXY_ADMIN's view on a non-fresh DB."""
    hashes: set = set()
    page = 1
    while True:
        resp = await proxy_client.get(
            f"/key/list?page={page}&size=100",
            headers={"Authorization": f"Bearer {caller_cleartext}"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        for entry in body.get("keys", []):
            tok = entry.get("token") if isinstance(entry, dict) else entry
            if tok:
                hashes.add(tok)
        if page >= (body.get("total_pages") or 1):
            return hashes
        page += 1


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

    returned_hashes = await _all_visible_hashes(proxy_client, caller.cleartext)
    visible_seeded = {
        hashed_to_actor[h] for h in returned_hashes if h in hashed_to_actor
    }
    assert visible_seeded == set(expected_visible), (
        f"{actor.value}: expected {sorted(a.value for a in expected_visible)}, "
        f"got {sorted(a.value for a in visible_seeded)}"
    )
