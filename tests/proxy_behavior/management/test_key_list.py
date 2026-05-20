"""Slice 9 — actor visibility matrix for ``GET /key/list``.

For ``/key/list`` the *response itself* is the matrix: each actor calls the
endpoint with default filters and we assert which seeded actor keys end up in
the returned set. The set-equality assertion is filtered to seeded tokens
only, so unrelated rows in the DB (other tests, leftover dev data) can't flap
the matrix.

8 scenarios — one per actor. Expected visibility is pinned against the
current handler so future changes to the filter logic surface red.
"""

from typing import FrozenSet

import pytest

from .actors import Actor

pytestmark = pytest.mark.asyncio(loop_scope="session")


# Maps each actor → the set of seeded actors whose keys it is permitted to see
# under default ``/key/list`` (no filter params, page=1, size=10).
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
    actor: Actor,
    expected_visible: FrozenSet[Actor],
    proxy_client,
    world,
):
    caller = world.keys[actor]
    seeded_hashes = {a: world.keys[a].hashed for a in Actor}
    hashed_to_actor = {h: a for a, h in seeded_hashes.items()}

    # Use size=100 to ensure we see all 8 seeded keys for proxy_admin.
    resp = await proxy_client.get(
        "/key/list?size=100",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
    )
    assert (
        resp.status_code == 200
    ), f"{actor.value} GET /key/list → {resp.status_code}: {resp.text}"

    body = resp.json()
    returned = body.get("keys", [])

    # /key/list can return either token strings (default) or full objects.
    # Default: list of dicts with ``token`` key. Reduce to hashes.
    returned_hashes = set()
    for entry in returned:
        if isinstance(entry, dict):
            tok = entry.get("token")
        else:
            tok = entry
        if tok:
            returned_hashes.add(tok)

    visible_seeded = {
        hashed_to_actor[h] for h in returned_hashes if h in hashed_to_actor
    }
    expected = set(expected_visible)
    assert visible_seeded == expected, (
        f"{actor.value} /key/list visibility differs:\n"
        f"  expected: {sorted(a.value for a in expected)}\n"
        f"  actual:   {sorted(a.value for a in visible_seeded)}\n"
        f"  diff (missing): {sorted(a.value for a in (expected - visible_seeded))}\n"
        f"  diff (extra):   {sorted(a.value for a in (visible_seeded - expected))}"
    )
