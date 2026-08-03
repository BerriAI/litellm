import uuid

import pytest

from litellm.proxy.utils import hash_token

from .actors import TEAM_ALPHA, TEAM_BETA, Actor
from .conftest import create_scratch_key

pytestmark = pytest.mark.asyncio(loop_scope="session")


# (id, actor, target_shape, expected_status). Pinned against current gating:
# proxy_admin bypasses; org_admin is blocked by an early role gate (401);
# every other (INTERNAL_USER-roled) actor hits user_id-mismatch 403, no-team-
# admin 403, or team_member_permission 401 depending on target / membership.
_SCENARIOS = [
    ("self/proxy_admin", Actor.PROXY_ADMIN, "self", 200),
    ("self/org_admin", Actor.ORG_ADMIN, "self", 401),
    ("self/team_admin", Actor.TEAM_ADMIN, "self", 403),
    ("self/internal_user", Actor.INTERNAL_USER, "self", 403),
    ("self/owner", Actor.OWNER, "self", 403),
    ("self/unrelated_same_org", Actor.UNRELATED_SAME_ORG, "self", 403),
    ("self/cross_org_user", Actor.CROSS_ORG_USER, "self", 403),
    ("self/service_account", Actor.SERVICE_ACCOUNT, "self", 403),
    ("owner_target/proxy_admin", Actor.PROXY_ADMIN, "owner", 200),
    ("owner_target/org_admin", Actor.ORG_ADMIN, "owner", 401),
    ("owner_target/team_admin", Actor.TEAM_ADMIN, "owner", 403),
    ("owner_target/internal_user", Actor.INTERNAL_USER, "owner", 403),
    ("owner_target/unrelated_same_org", Actor.UNRELATED_SAME_ORG, "owner", 403),
    ("owner_target/cross_org_user", Actor.CROSS_ORG_USER, "owner", 403),
    ("owner_target/service_account", Actor.SERVICE_ACCOUNT, "owner", 403),
    ("cross_org_target/proxy_admin", Actor.PROXY_ADMIN, "cross_org", 200),
    ("cross_org_target/org_admin", Actor.ORG_ADMIN, "cross_org", 401),
    ("cross_org_target/team_admin", Actor.TEAM_ADMIN, "cross_org", 403),
    ("cross_org_target/owner", Actor.OWNER, "cross_org", 403),
    ("cross_org_target/cross_org_user", Actor.CROSS_ORG_USER, "cross_org", 401),
    ("cross_org_target/service_account", Actor.SERVICE_ACCOUNT, "cross_org", 403),
]

MARKER_MODEL = "behavior-pin-update-marker-model"


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
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext

    if target_shape == "self":
        target_cleartext = await create_scratch_key(
            proxy_client, seeder, scratch.prefix, user_id=caller.user_id
        )
    elif target_shape == "owner":
        target_cleartext = await create_scratch_key(
            proxy_client,
            seeder,
            scratch.prefix,
            user_id=world.keys[Actor.OWNER].user_id,
            team_id=TEAM_ALPHA,
        )
    elif target_shape == "cross_org":
        target_cleartext = await create_scratch_key(
            proxy_client,
            seeder,
            scratch.prefix,
            user_id=world.keys[Actor.CROSS_ORG_USER].user_id,
            team_id=TEAM_BETA,
        )
    else:
        pytest.fail(f"unknown target_shape={target_shape}")

    target_hashed = hash_token(target_cleartext)

    resp = await proxy_client.post(
        "/key/update",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={"key": target_cleartext, "models": [MARKER_MODEL]},
    )
    assert (
        resp.status_code == expected_status
    ), f"{actor.value} {target_shape}: {resp.status_code} {resp.text}"

    row = await prisma.db.litellm_verificationtoken.find_unique(
        where={"token": target_hashed}
    )
    assert row is not None
    if expected_status == 200:
        assert row.models == [MARKER_MODEL]
    else:
        assert row.models != [MARKER_MODEL], "denied but row mutated"


async def _seed_shape(proxy_client, seeder, prefix, world, shape, caller) -> str:
    if shape == "self":
        return await create_scratch_key(
            proxy_client, seeder, prefix, user_id=caller.user_id
        )
    if shape == "owner":
        return await create_scratch_key(
            proxy_client,
            seeder,
            prefix,
            user_id=world.keys[Actor.OWNER].user_id,
            team_id=TEAM_ALPHA,
        )
    if shape == "cross_org":
        return await create_scratch_key(
            proxy_client,
            seeder,
            prefix,
            user_id=world.keys[Actor.CROSS_ORG_USER].user_id,
            team_id=TEAM_BETA,
        )
    pytest.fail(f"unknown shape={shape}")  # pragma: no cover


async def test_key_update_missing_key_is_404(proxy_client, world):
    """An update targeting a key absent from the DB is a 404 — not 401/403."""
    resp = await proxy_client.post(
        "/key/update",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"key": "sk-" + uuid.uuid4().hex, "models": [MARKER_MODEL]},
    )
    assert resp.status_code == 404, resp.text


# A denied /key/update must not partially apply: the budget/limit columns are
# left untouched. Each scenario is a denial cell from the matrix above.
_DENIED_BUDGET = [
    ("team_admin/self", Actor.TEAM_ADMIN, "self", 403),
    ("internal_user/owner", Actor.INTERNAL_USER, "owner", 403),
    ("cross_org_user/cross_org", Actor.CROSS_ORG_USER, "cross_org", 401),
]


@pytest.mark.parametrize(
    "actor,target_shape,expected_status",
    [(a, t, s) for (_id, a, t, s) in _DENIED_BUDGET],
    ids=[s[0] for s in _DENIED_BUDGET],
)
async def test_key_update_denied_does_not_touch_budget_counters(
    actor: Actor,
    target_shape: str,
    expected_status: int,
    proxy_client,
    prisma,
    scratch,
    world,
):
    caller = world.keys[actor]
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    target = await _seed_shape(
        proxy_client, seeder, scratch.prefix, world, target_shape, caller
    )
    target_hashed = hash_token(target)

    resp = await proxy_client.post(
        "/key/update",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={"key": target, "max_budget": 999.0, "tpm_limit": 888, "rpm_limit": 777},
    )
    assert (
        resp.status_code == expected_status
    ), f"{actor.value} {target_shape}: {resp.status_code} {resp.text}"

    row = await prisma.db.litellm_verificationtoken.find_unique(
        where={"token": target_hashed}
    )
    assert row is not None
    assert row.max_budget is None, "denied but max_budget applied"
    assert row.tpm_limit is None, "denied but tpm_limit applied"
    assert row.rpm_limit is None, "denied but rpm_limit applied"
