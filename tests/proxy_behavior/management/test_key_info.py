"""Slice 8 — actor × target authz matrix for ``GET /key/info``.

Targets are three fixed seeded keys representing the canonical relations:

  * **own** — the actor's own key (each actor's relation to itself).
  * **owner_key** — OWNER actor's key in org_a / team_alpha. From any
    other org_a / team_alpha actor's perspective this is "same-team, not
    mine". From cross_org_user's perspective this is cross-org.
  * **cross_org_user_key** — CROSS_ORG_USER's key in org_b / team_beta.
    From any org_a actor's perspective this is cross-org.

24 (actor × target) scenarios after applying the matrix; status codes pinned
against current handler behavior so future PRs that change the visibility
boundary turn the suite red.
"""

import pytest

from .actors import Actor
from .conftest import MASTER_KEY  # noqa: F401  (kept for symmetry with sibling files)

pytestmark = pytest.mark.asyncio(loop_scope="session")


# (id, actor, target_actor, expected_status)
_SCENARIOS = [
    # ─── target = own key ─────────────────────────────────────────────────
    ("own/proxy_admin", Actor.PROXY_ADMIN, Actor.PROXY_ADMIN, 200),
    ("own/org_admin", Actor.ORG_ADMIN, Actor.ORG_ADMIN, 200),
    ("own/team_admin", Actor.TEAM_ADMIN, Actor.TEAM_ADMIN, 200),
    ("own/internal_user", Actor.INTERNAL_USER, Actor.INTERNAL_USER, 200),
    ("own/owner", Actor.OWNER, Actor.OWNER, 200),
    ("own/unrelated_same_org", Actor.UNRELATED_SAME_ORG, Actor.UNRELATED_SAME_ORG, 200),
    ("own/cross_org_user", Actor.CROSS_ORG_USER, Actor.CROSS_ORG_USER, 200),
    ("own/service_account", Actor.SERVICE_ACCOUNT, Actor.SERVICE_ACCOUNT, 200),
    # ─── target = OWNER's key (org_a / team_alpha) ────────────────────────
    # NB: org_admin currently 403s on individual key info even within their own
    # org — visibility is scoped to "your own keys" + "your team's keys", not
    # "your org's keys". Same-team peers (internal_user, unrelated_same_org,
    # service_account) DO see each other's keys.
    ("owner_key/proxy_admin", Actor.PROXY_ADMIN, Actor.OWNER, 200),
    ("owner_key/org_admin", Actor.ORG_ADMIN, Actor.OWNER, 403),
    ("owner_key/team_admin", Actor.TEAM_ADMIN, Actor.OWNER, 200),
    ("owner_key/internal_user", Actor.INTERNAL_USER, Actor.OWNER, 200),
    ("owner_key/owner", Actor.OWNER, Actor.OWNER, 200),
    ("owner_key/unrelated_same_org", Actor.UNRELATED_SAME_ORG, Actor.OWNER, 200),
    ("owner_key/cross_org_user", Actor.CROSS_ORG_USER, Actor.OWNER, 403),
    ("owner_key/service_account", Actor.SERVICE_ACCOUNT, Actor.OWNER, 200),
    # ─── target = CROSS_ORG_USER's key (org_b / team_beta) ────────────────
    ("cross_org/proxy_admin", Actor.PROXY_ADMIN, Actor.CROSS_ORG_USER, 200),
    ("cross_org/org_admin", Actor.ORG_ADMIN, Actor.CROSS_ORG_USER, 403),
    ("cross_org/team_admin", Actor.TEAM_ADMIN, Actor.CROSS_ORG_USER, 403),
    ("cross_org/internal_user", Actor.INTERNAL_USER, Actor.CROSS_ORG_USER, 403),
    ("cross_org/owner", Actor.OWNER, Actor.CROSS_ORG_USER, 403),
    (
        "cross_org/unrelated_same_org",
        Actor.UNRELATED_SAME_ORG,
        Actor.CROSS_ORG_USER,
        403,
    ),
    ("cross_org/cross_org_user", Actor.CROSS_ORG_USER, Actor.CROSS_ORG_USER, 200),
    ("cross_org/service_account", Actor.SERVICE_ACCOUNT, Actor.CROSS_ORG_USER, 403),
]


@pytest.mark.parametrize(
    "actor,target_actor,expected_status",
    [(a, t, s) for (_id, a, t, s) in _SCENARIOS],
    ids=[s[0] for s in _SCENARIOS],
)
async def test_key_info_authz_matrix(
    actor: Actor,
    target_actor: Actor,
    expected_status: int,
    proxy_client,
    world,
):
    caller = world.keys[actor]
    target = world.keys[target_actor]

    resp = await proxy_client.get(
        f"/key/info?key={target.cleartext}",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
    )
    assert resp.status_code == expected_status, (
        f"{actor.value} GET /key/info?key=<{target_actor.value}> → "
        f"{resp.status_code} (expected {expected_status}). body={resp.text}"
    )

    if expected_status == 200:
        body = resp.json()
        # The handler echoes back whatever ``key`` was passed in the query string,
        # so body["key"] == cleartext when we querystring it. When the auth key
        # IS the target (no query), it returns the hashed form. Either match is
        # fine — the canonical identity check is on info.user_id.
        assert body.get("key") in (target.cleartext, target.hashed), (
            f"{actor.value} → target {target_actor.value}: wrong key in response "
            f"(got {body.get('key')!r}, expected cleartext or hashed of target)"
        )
        assert body["info"].get("user_id") == target.user_id, (
            f"{actor.value} → target {target_actor.value}: wrong user_id "
            f"(got {body['info'].get('user_id')!r}, expected {target.user_id!r})"
        )
