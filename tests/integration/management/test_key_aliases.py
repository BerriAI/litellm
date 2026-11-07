import uuid
from typing import FrozenSet

import pytest

from litellm.proxy.utils import hash_token

from tests.integration._support.actors import TEAM_ALPHA, TEAM_BETA, Actor

pytestmark = pytest.mark.asyncio(loop_scope="session")


# GET /key/aliases scopes non-admins via _apply_non_admin_alias_scope: a
# non-admin sees an alias only if it owns the key (user_id match) or the key
# belongs to one of its teams. PROXY_ADMIN sees every alias. The seeded keys:
#   own   — owned by INTERNAL_USER, no team  -> user_id scope only
#   alpha — owned by OWNER, team TEAM_ALPHA  -> team scope for alpha members
#   beta  — owned by CROSS_ORG_USER, TEAM_BETA
async def _seed_alias_keys(db, prefix: str, actors) -> dict:
    spec = {
        "own": (Actor.INTERNAL_USER, None),
        "alpha": (Actor.OWNER, actors.team_alpha_id),
        "beta": (Actor.CROSS_ORG_USER, actors.team_beta_id),
    }
    out = {}
    for tag, (owner, team_id) in spec.items():
        alias = f"{prefix}-{tag}"
        data = {
            "token": hash_token("sk-" + uuid.uuid4().hex),
            "key_name": f"{prefix}-{tag}-key",
            "key_alias": alias,
            "user_id": actors.keys[owner].user_id,
            "models": [],
        }
        if team_id is not None:
            data["team_id"] = team_id
        await db.db.litellm_verificationtoken.create(data=data)
        out[tag] = alias
    return out


async def _fetch_aliases(gateway_client, caller_cleartext: str, query: str) -> set:
    resp = await gateway_client.get(
        f"/key/aliases?{query}&size=100",
        headers={"Authorization": f"Bearer {caller_cleartext}"},
    )
    assert resp.status_code == 200, resp.text
    return set(resp.json()["aliases"])


# ORG_ADMIN-role callers are stopped 401 by the management-route gate before
# the handler runs — /key/aliases carries no org context. Every other actor
# reaches the handler and is scoped by _apply_non_admin_alias_scope.
_VISIBILITY = {
    Actor.PROXY_ADMIN: (200, frozenset({"own", "alpha", "beta"})),
    Actor.ORG_ADMIN: (401, None),
    Actor.TEAM_ADMIN: (200, frozenset({"alpha"})),
    Actor.INTERNAL_USER: (200, frozenset({"own", "alpha"})),
    Actor.OWNER: (200, frozenset({"alpha"})),
    Actor.UNRELATED_SAME_ORG: (200, frozenset({"alpha"})),
    Actor.CROSS_ORG_USER: (200, frozenset({"beta"})),
    Actor.SERVICE_ACCOUNT: (200, frozenset({"alpha"})),
    Actor.ORG_B_ADMIN: (401, None),
}


@pytest.mark.parametrize(
    "actor,expected_status,expected_tags",
    [(a, s, t) for a, (s, t) in _VISIBILITY.items()],
    ids=[a.value for a in _VISIBILITY],
)
async def test_key_aliases_visibility(
    actor: Actor,
    expected_status: int,
    expected_tags: FrozenSet[str],
    gateway_client,
    db,
    resources,
    actors,
):
    aliases = await _seed_alias_keys(db, resources.prefix, actors)
    known = {v: k for k, v in aliases.items()}

    resp = await gateway_client.get(
        f"/key/aliases?search={resources.prefix}&size=100",
        headers={"Authorization": f"Bearer {actors.keys[actor].cleartext}"},
    )
    assert (
        resp.status_code == expected_status
    ), f"{actor.value}: {resp.status_code} {resp.text}"
    if expected_status != 200:
        return

    visible = {known[a] for a in resp.json()["aliases"] if a in known}
    assert visible == set(
        expected_tags
    ), f"{actor.value}: expected {sorted(expected_tags)}, got {sorted(visible)}"


async def test_key_aliases_team_id_filter(gateway_client, db, resources, actors):
    """team_id filter narrows the result to keys of that team."""
    aliases = await _seed_alias_keys(db, resources.prefix, actors)
    returned = await _fetch_aliases(
        gateway_client,
        actors.keys[Actor.PROXY_ADMIN].cleartext,
        f"search={resources.prefix}&team_id={actors.team_alpha_id}",
    )
    assert returned & set(aliases.values()) == {aliases["alpha"]}


async def test_key_aliases_search_filter(gateway_client, db, resources, actors):
    """search is a case-insensitive substring match on key_alias."""
    aliases = await _seed_alias_keys(db, resources.prefix, actors)
    returned = await _fetch_aliases(
        gateway_client,
        actors.keys[Actor.PROXY_ADMIN].cleartext,
        f"search={aliases['beta']}",
    )
    assert returned & set(aliases.values()) == {aliases["beta"]}
