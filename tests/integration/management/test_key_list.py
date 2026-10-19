from typing import FrozenSet

import pytest

from litellm.proxy.utils import hash_token

from tests.integration._support.actors import TEAM_ALPHA, Actor
from tests.integration._support.actors import create_scratch_key

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


async def _all_visible_hashes(gateway_client, caller_cleartext) -> set:
    """Walk every /key/list page — size is capped at 100 by the endpoint, so a
    single request can truncate PROXY_ADMIN's view on a non-fresh DB."""
    hashes: set = set()
    page = 1
    while True:
        resp = await gateway_client.get(
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
    actor: Actor, expected_visible: FrozenSet[Actor], gateway_client, actors
):
    caller = actors.keys[actor]
    hashed_to_actor = {actors.keys[a].hashed: a for a in Actor}

    returned_hashes = await _all_visible_hashes(gateway_client, caller.cleartext)
    visible_seeded = {
        hashed_to_actor[h] for h in returned_hashes if h in hashed_to_actor
    }
    assert visible_seeded == set(expected_visible), (
        f"{actor.value}: expected {sorted(a.value for a in expected_visible)}, "
        f"got {sorted(a.value for a in visible_seeded)}"
    )


async def _list_hashes(gateway_client, caller_cleartext: str, query: str) -> set:
    resp = await gateway_client.get(
        f"/key/list?{query}&size=100",
        headers={"Authorization": f"Bearer {caller_cleartext}"},
    )
    assert resp.status_code == 200, resp.text
    hashes: set = set()
    for entry in resp.json().get("keys", []):
        tok = entry.get("token") if isinstance(entry, dict) else entry
        if tok:
            hashes.add(tok)
    return hashes


async def test_key_list_admin_key_alias_substring_match(gateway_client, resources, actors):
    """A PROXY_ADMIN's key_alias filter is a case-insensitive substring match
    when substring_matching=true is requested (the dashboard search box); a
    narrower fragment selects the subset whose alias contains it. Substring
    matching is opt-in: without the flag the filter is exact (see
    test_key_list_admin_key_alias_exact_without_substring_flag)."""
    admin = actors.keys[Actor.PROXY_ADMIN]
    a = await create_scratch_key(
        gateway_client,
        admin.cleartext,
        resources.prefix,
        user_id=admin.user_id,
        key_alias=f"{resources.prefix}-sub-a",
    )
    b = await create_scratch_key(
        gateway_client,
        admin.cleartext,
        resources.prefix,
        user_id=admin.user_id,
        key_alias=f"{resources.prefix}-sub-b",
    )
    seeded = {hash_token(a), hash_token(b)}

    broad = await _list_hashes(
        gateway_client,
        admin.cleartext,
        f"key_alias={resources.prefix}-sub&substring_matching=true",
    )
    assert broad & seeded == seeded

    narrow = await _list_hashes(
        gateway_client,
        admin.cleartext,
        f"key_alias={resources.prefix}-sub-a&substring_matching=true",
    )
    assert narrow & seeded == {hash_token(a)}


async def test_key_list_admin_key_alias_exact_without_substring_flag(
    gateway_client, resources, actors
):
    """Regression guard for the prior exact-match contract: without
    substring_matching, even a PROXY_ADMIN's key_alias filter is exact, so a
    fragment of a seeded alias does not select it."""
    admin = actors.keys[Actor.PROXY_ADMIN]
    full_alias = f"{resources.prefix}-exactflag"
    key = await create_scratch_key(
        gateway_client,
        admin.cleartext,
        resources.prefix,
        user_id=admin.user_id,
        key_alias=full_alias,
    )
    key_hash = hash_token(key)

    exact = await _list_hashes(gateway_client, admin.cleartext, f"key_alias={full_alias}")
    assert key_hash in exact

    fragment = await _list_hashes(
        gateway_client, admin.cleartext, f"key_alias={resources.prefix}-exactfla"
    )
    assert key_hash not in fragment


async def test_key_list_non_admin_key_alias_is_exact_match(
    gateway_client, resources, actors
):
    """A non-admin's key_alias filter is exact-match only — substring filtering
    is restricted to admins. The full alias matches; a fragment does not."""
    caller = actors.keys[Actor.INTERNAL_USER]
    alias = f"{resources.prefix}-exact"
    key = await create_scratch_key(
        gateway_client,
        actors.keys[Actor.PROXY_ADMIN].cleartext,
        resources.prefix,
        user_id=caller.user_id,
        key_alias=alias,
    )
    key_hash = hash_token(key)

    exact = await _list_hashes(gateway_client, caller.cleartext, f"key_alias={alias}")
    assert key_hash in exact

    fragment = await _list_hashes(
        gateway_client, caller.cleartext, f"key_alias={resources.prefix}-exac"
    )
    assert key_hash not in fragment


async def test_key_list_team_id_filter(gateway_client, resources, actors):
    """A team_id filter narrows the listing to keys of that team."""
    admin = actors.keys[Actor.PROXY_ADMIN]
    team_key = await create_scratch_key(
        gateway_client,
        admin.cleartext,
        resources.prefix,
        user_id=actors.keys[Actor.OWNER].user_id,
        team_id=actors.team_alpha_id,
        key_alias=f"{resources.prefix}-team",
    )
    no_team_key = await create_scratch_key(
        gateway_client,
        admin.cleartext,
        resources.prefix,
        user_id=admin.user_id,
        key_alias=f"{resources.prefix}-noteam",
    )

    hashes = await _list_hashes(gateway_client, admin.cleartext, f"team_id={actors.team_alpha_id}")
    assert hash_token(team_key) in hashes
    assert hash_token(no_team_key) not in hashes


async def test_key_list_non_admin_cannot_filter_other_team(gateway_client, actors):
    """A non-admin filtering by a team it does not belong to is rejected 403."""
    resp = await gateway_client.get(
        f"/key/list?team_id={actors.team_beta_id}",
        headers={
            "Authorization": f"Bearer {actors.keys[Actor.INTERNAL_USER].cleartext}"
        },
    )
    assert resp.status_code == 403, resp.text
