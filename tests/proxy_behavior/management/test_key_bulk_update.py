import uuid

import pytest

from litellm.proxy.utils import hash_token

from .actors import Actor
from .conftest import create_scratch_key

pytestmark = pytest.mark.asyncio(loop_scope="session")

_MARKER_BUDGET = 42.0


# POST /key/bulk_update is PROXY_ADMIN-only. The handler's own gate is
# user_role != PROXY_ADMIN -> 403, but ORG_ADMIN-role callers never reach it:
# the management-route gate 401s them first (the body carries no org context,
# and /key/bulk_update is an internal_user route, not an org-admin one).
# INTERNAL_USER-role callers clear the route gate and hit the handler's 403.
_MATRIX = [
    ("proxy_admin", Actor.PROXY_ADMIN, 200),
    ("org_admin", Actor.ORG_ADMIN, 401),
    ("team_admin", Actor.TEAM_ADMIN, 403),
    ("internal_user", Actor.INTERNAL_USER, 403),
    ("owner", Actor.OWNER, 403),
    ("unrelated_same_org", Actor.UNRELATED_SAME_ORG, 403),
    ("cross_org_user", Actor.CROSS_ORG_USER, 403),
    ("service_account", Actor.SERVICE_ACCOUNT, 403),
]


@pytest.mark.parametrize(
    "actor,expected_status",
    [(a, s) for (_id, a, s) in _MATRIX],
    ids=[s[0] for s in _MATRIX],
)
async def test_key_bulk_update_authz_matrix(
    actor: Actor,
    expected_status: int,
    proxy_client,
    prisma,
    scratch,
    world,
):
    caller = world.keys[actor]
    seeder = world.keys[Actor.PROXY_ADMIN].cleartext
    target = await create_scratch_key(
        proxy_client, seeder, scratch.prefix, user_id=caller.user_id
    )
    hashed = hash_token(target)

    resp = await proxy_client.post(
        "/key/bulk_update",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={"keys": [{"key": target, "max_budget": _MARKER_BUDGET}]},
    )
    assert (
        resp.status_code == expected_status
    ), f"{actor.value}: {resp.status_code} {resp.text}"

    row = await prisma.db.litellm_verificationtoken.find_unique(where={"token": hashed})
    assert row is not None
    if expected_status == 200:
        body = resp.json()
        assert len(body["successful_updates"]) == 1
        assert body["failed_updates"] == []
        assert row.max_budget == _MARKER_BUDGET
    else:
        assert row.max_budget != _MARKER_BUDGET, "denied but key mutated"


async def test_key_bulk_update_empty_keys_is_400(proxy_client, world):
    """An empty batch is rejected 400 before any per-key processing."""
    resp = await proxy_client.post(
        "/key/bulk_update",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"keys": []},
    )
    assert resp.status_code == 400, resp.text


async def test_key_bulk_update_over_max_batch_is_400(proxy_client, world):
    """A batch larger than the 500-key cap is rejected 400."""
    items = [{"key": "sk-" + uuid.uuid4().hex} for _ in range(501)]
    resp = await proxy_client.post(
        "/key/bulk_update",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"keys": items},
    )
    assert resp.status_code == 400, resp.text


async def test_key_bulk_update_per_key_failure_is_isolated(
    proxy_client, prisma, scratch, world
):
    """One bad key in the batch does not abort the others — it lands in
    failed_updates while the valid key is still updated."""
    admin = world.keys[Actor.PROXY_ADMIN]
    valid = await create_scratch_key(
        proxy_client, admin.cleartext, scratch.prefix, user_id=admin.user_id
    )
    missing = "sk-" + uuid.uuid4().hex

    resp = await proxy_client.post(
        "/key/bulk_update",
        headers={"Authorization": f"Bearer {admin.cleartext}"},
        json={
            "keys": [
                {"key": valid, "max_budget": _MARKER_BUDGET},
                {"key": missing, "max_budget": _MARKER_BUDGET},
            ]
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_requested"] == 2
    assert len(body["successful_updates"]) == 1
    assert len(body["failed_updates"]) == 1

    row = await prisma.db.litellm_verificationtoken.find_unique(
        where={"token": hash_token(valid)}
    )
    assert row is not None and row.max_budget == _MARKER_BUDGET
