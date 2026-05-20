"""Slice 10 — actor × target authz matrix for ``POST /key/update``.

Each scenario:
  1. The master key seeds a fresh scratch key with a specific (user_id,
     team_id, organization_id) scope on it.
  2. The actor under test calls ``POST /key/update`` to flip its ``models``
     field to a known marker list.
  3. The test asserts the status code and (when 200) that the mutation
     actually landed in the DB.

Three target shapes:
  * **self**: target is scoped to the actor's own user (the actor's "own" key).
  * **owner_target**: scoped to OWNER in org_a / team_alpha — the same-team /
    same-org / different-user target.
  * **cross_org_target**: scoped to CROSS_ORG_USER in org_b / team_beta — the
    cross-org target.

Status codes are pinned against observed handler behavior.
"""

from typing import Any, Dict, List

import pytest

from .actors import TEAM_ALPHA, TEAM_BETA, Actor
from .conftest import MASTER_KEY

pytestmark = pytest.mark.asyncio(loop_scope="session")


# (id, actor, target_shape, expected_status)
#
# Pinned against current /key/update gating:
#
#   * PROXY_ADMIN bypasses every check (200).
#   * ORG_ADMIN is blocked by an early role gate, regardless of target
#     scope, with a 401 "Only proxy admin..." auth_error.
#   * Every other (INTERNAL_USER-rolesed) actor hits one of:
#     - 403 "User can only create keys for themselves" when the target key's
#       user_id is not theirs;
#     - 403 "Only proxy admins, team admins, or org admins can call /key/update"
#       when they own the target but lack team-admin status on the target
#       key's team (or the key has no team);
#     - 401 "Team member does not have permissions" when they own the target
#       in a team they're a member of without admin role.
_SCENARIOS = [
    # ─── target = self-owned key (no team_id on the key) ──────────────────
    ("self/proxy_admin", Actor.PROXY_ADMIN, "self", 200),
    ("self/org_admin", Actor.ORG_ADMIN, "self", 401),
    ("self/team_admin", Actor.TEAM_ADMIN, "self", 403),
    ("self/internal_user", Actor.INTERNAL_USER, "self", 403),
    ("self/owner", Actor.OWNER, "self", 403),
    ("self/unrelated_same_org", Actor.UNRELATED_SAME_ORG, "self", 403),
    ("self/cross_org_user", Actor.CROSS_ORG_USER, "self", 403),
    ("self/service_account", Actor.SERVICE_ACCOUNT, "self", 403),
    # ─── target = OWNER-scoped key in org_a / team_alpha ──────────────────
    ("owner_target/proxy_admin", Actor.PROXY_ADMIN, "owner", 200),
    ("owner_target/org_admin", Actor.ORG_ADMIN, "owner", 401),
    # team_admin gets 403 "user can only create keys for themselves" — the
    # user_id-mismatch check fires BEFORE the team-admin check.
    ("owner_target/team_admin", Actor.TEAM_ADMIN, "owner", 403),
    ("owner_target/internal_user", Actor.INTERNAL_USER, "owner", 403),
    ("owner_target/unrelated_same_org", Actor.UNRELATED_SAME_ORG, "owner", 403),
    ("owner_target/cross_org_user", Actor.CROSS_ORG_USER, "owner", 403),
    ("owner_target/service_account", Actor.SERVICE_ACCOUNT, "owner", 403),
    # ─── target = CROSS_ORG_USER-scoped key in org_b / team_beta ──────────
    ("cross_org_target/proxy_admin", Actor.PROXY_ADMIN, "cross_org", 200),
    ("cross_org_target/org_admin", Actor.ORG_ADMIN, "cross_org", 401),
    ("cross_org_target/team_admin", Actor.TEAM_ADMIN, "cross_org", 403),
    ("cross_org_target/owner", Actor.OWNER, "cross_org", 403),
    # cross_org_user owns the target (user_id matches) and is in team_beta as
    # a non-admin → team_member_permission_error 401.
    ("cross_org_target/cross_org_user", Actor.CROSS_ORG_USER, "cross_org", 401),
    ("cross_org_target/service_account", Actor.SERVICE_ACCOUNT, "cross_org", 403),
]

MARKER_MODEL = "behavior-pin-update-marker-model"


async def _create_scratch_key(
    proxy_client,
    prisma,
    scratch_prefix: str,
    *,
    user_id: str,
    team_id: str = None,
    organization_id: str = None,
) -> str:
    """Seed a fresh key tagged with scratch_prefix and return its cleartext."""
    body: Dict[str, Any] = {"key_alias": scratch_prefix, "user_id": user_id}
    if team_id is not None:
        body["team_id"] = team_id
    if organization_id is not None:
        body["organization_id"] = organization_id
    resp = await proxy_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {MASTER_KEY}"},
        json=body,
    )
    assert resp.status_code == 200, f"setup: master /key/generate failed: {resp.text}"
    return resp.json()["key"]


@pytest.mark.parametrize(
    "actor,target_shape,expected_status",
    [(a, t, s) for (_id, a, t, s) in _SCENARIOS],
    ids=[s[0] for s in _SCENARIOS],
)
async def test_key_update_authz_matrix(
    actor: Actor,
    target_shape: str,
    expected_status: int,
    proxy_client,
    prisma,
    scratch,
    world,
):
    caller = world.keys[actor]

    if target_shape == "self":
        target_cleartext = await _create_scratch_key(
            proxy_client, prisma, scratch.prefix, user_id=caller.user_id
        )
    elif target_shape == "owner":
        target_cleartext = await _create_scratch_key(
            proxy_client,
            prisma,
            scratch.prefix,
            user_id=world.keys[Actor.OWNER].user_id,
            team_id=TEAM_ALPHA,
        )
    elif target_shape == "cross_org":
        target_cleartext = await _create_scratch_key(
            proxy_client,
            prisma,
            scratch.prefix,
            user_id=world.keys[Actor.CROSS_ORG_USER].user_id,
            team_id=TEAM_BETA,
        )
    else:
        pytest.fail(f"unknown target_shape={target_shape}")

    from litellm.proxy.utils import hash_token

    target_hashed = hash_token(target_cleartext)

    resp = await proxy_client.post(
        "/key/update",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={"key": target_cleartext, "models": [MARKER_MODEL]},
    )
    assert resp.status_code == expected_status, (
        f"{actor.value} POST /key/update {target_shape} → {resp.status_code} "
        f"(expected {expected_status}). body={resp.text}"
    )

    row = await prisma.db.litellm_verificationtoken.find_unique(
        where={"token": target_hashed}
    )
    assert row is not None, "scratch key vanished mid-test"

    if expected_status == 200:
        assert row.models == [MARKER_MODEL], (
            f"{actor.value}: handler returned 200 but row.models did not flip — "
            f"got {row.models!r}"
        )
    else:
        assert row.models != [MARKER_MODEL], (
            f"{actor.value}: handler returned {expected_status} but row.models was "
            f"mutated to {row.models!r}"
        )
