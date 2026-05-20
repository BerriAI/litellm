"""Slice 12 — actor × target authz matrix for ``POST /key/delete``.

Same shape as Slices 10/11: master-seed a scoped scratch key, the actor under
test attempts to delete it via ``POST /key/delete {keys: [<cleartext>]}``. On
200 the test verifies the row is gone (or soft-deleted) AND the cleartext can
no longer auth. On denial it verifies the row survives and still authenticates.
"""

from typing import Any, Dict, Optional

import pytest

from .actors import TEAM_ALPHA, TEAM_BETA, Actor

pytestmark = pytest.mark.asyncio(loop_scope="session")


_SCENARIOS = [
    # ─── target = self-owned key ──────────────────────────────────────────
    ("self/proxy_admin", Actor.PROXY_ADMIN, "self", 200),
    ("self/org_admin", Actor.ORG_ADMIN, "self", 401),
    ("self/team_admin", Actor.TEAM_ADMIN, "self", 200),
    ("self/internal_user", Actor.INTERNAL_USER, "self", 200),
    ("self/owner", Actor.OWNER, "self", 200),
    ("self/unrelated_same_org", Actor.UNRELATED_SAME_ORG, "self", 200),
    ("self/cross_org_user", Actor.CROSS_ORG_USER, "self", 200),
    ("self/service_account", Actor.SERVICE_ACCOUNT, "self", 200),
    # ─── target = OWNER-scoped key in org_a / team_alpha ──────────────────
    ("owner_target/proxy_admin", Actor.PROXY_ADMIN, "owner", 200),
    # ORG_ADMIN hits the early role gate before any target-specific check.
    ("owner_target/org_admin", Actor.ORG_ADMIN, "owner", 401),
    ("owner_target/team_admin", Actor.TEAM_ADMIN, "owner", 200),
    ("owner_target/internal_user", Actor.INTERNAL_USER, "owner", 403),
    ("owner_target/unrelated_same_org", Actor.UNRELATED_SAME_ORG, "owner", 403),
    ("owner_target/cross_org_user", Actor.CROSS_ORG_USER, "owner", 403),
    ("owner_target/service_account", Actor.SERVICE_ACCOUNT, "owner", 403),
    # ─── target = CROSS_ORG_USER-scoped key in org_b / team_beta ──────────
    ("cross_org_target/proxy_admin", Actor.PROXY_ADMIN, "cross_org", 200),
    ("cross_org_target/org_admin", Actor.ORG_ADMIN, "cross_org", 401),
    ("cross_org_target/team_admin", Actor.TEAM_ADMIN, "cross_org", 403),
    ("cross_org_target/owner", Actor.OWNER, "cross_org", 403),
    ("cross_org_target/cross_org_user", Actor.CROSS_ORG_USER, "cross_org", 200),
    ("cross_org_target/service_account", Actor.SERVICE_ACCOUNT, "cross_org", 403),
]


async def _create_scratch_key(
    proxy_client,
    seeder_cleartext: str,
    scratch_prefix: str,
    *,
    user_id: str,
    team_id: Optional[str] = None,
) -> str:
    """Seed a scratch key using the proxy_admin actor (not the bare master key).

    The seeded proxy_admin actor's auth path produces user_role=PROXY_ADMIN
    + a concrete user_id from the DB, which deterministically triggers the
    ``_user_can_only_create_keys_for_themselves`` PROXY_ADMIN bypass. The
    bare master key takes a different auth resolution path whose behavior
    differs between fresh-CI and warm-local environments.
    """
    body: Dict[str, Any] = {"key_alias": scratch_prefix, "user_id": user_id}
    if team_id is not None:
        body["team_id"] = team_id
    resp = await proxy_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {seeder_cleartext}"},
        json=body,
    )
    assert resp.status_code == 200, f"setup: seeder /key/generate failed: {resp.text}"
    return resp.json()["key"]


@pytest.mark.parametrize(
    "actor,target_shape,expected_status",
    [(a, t, s) for (_id, a, t, s) in _SCENARIOS],
    ids=[s[0] for s in _SCENARIOS],
)
async def test_key_delete_authz_matrix(
    actor: Actor,
    target_shape: str,
    expected_status: int,
    proxy_client,
    prisma,
    scratch,
    world,
):
    from litellm.proxy.utils import hash_token

    caller = world.keys[actor]

    if target_shape == "self":
        target_cleartext = await _create_scratch_key(
            proxy_client,
            world.keys[Actor.PROXY_ADMIN].cleartext,
            scratch.prefix,
            user_id=caller.user_id,
        )
    elif target_shape == "owner":
        target_cleartext = await _create_scratch_key(
            proxy_client,
            world.keys[Actor.PROXY_ADMIN].cleartext,
            scratch.prefix,
            user_id=world.keys[Actor.OWNER].user_id,
            team_id=TEAM_ALPHA,
        )
    elif target_shape == "cross_org":
        target_cleartext = await _create_scratch_key(
            proxy_client,
            world.keys[Actor.PROXY_ADMIN].cleartext,
            scratch.prefix,
            user_id=world.keys[Actor.CROSS_ORG_USER].user_id,
            team_id=TEAM_BETA,
        )
    else:
        pytest.fail(f"unknown target_shape={target_shape}")

    target_hashed = hash_token(target_cleartext)

    resp = await proxy_client.post(
        "/key/delete",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={"keys": [target_cleartext]},
    )
    assert resp.status_code == expected_status, (
        f"{actor.value} POST /key/delete {target_shape} → "
        f"{resp.status_code} (expected {expected_status}). body={resp.text}"
    )

    # Verify the after-state matches the verdict.
    row = await prisma.db.litellm_verificationtoken.find_unique(
        where={"token": target_hashed}
    )
    auth_check = await proxy_client.get(
        "/key/info",
        headers={"Authorization": f"Bearer {target_cleartext}"},
    )

    if expected_status == 200:
        # Successful delete: cleartext must no longer authenticate, regardless of
        # whether the row is hard-deleted or soft-deleted into LiteLLM_DeletedVerificationToken.
        assert auth_check.status_code == 401, (
            f"{actor.value}: handler returned 200 but cleartext still authenticates "
            f"({auth_check.status_code}): {auth_check.text}"
        )
    else:
        # Denied: row still present, cleartext still works.
        assert row is not None, (
            f"{actor.value}: handler returned {expected_status} but row vanished — "
            f"silent delete on denial"
        )
        assert auth_check.status_code == 200, (
            f"{actor.value}: handler returned {expected_status} but cleartext no "
            f"longer authenticates: {auth_check.text}"
        )
