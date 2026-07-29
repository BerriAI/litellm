import uuid

import pytest

from litellm.proxy.utils import hash_token

from .actors import TEAM_ALPHA, TEAM_BETA, Actor
from .conftest import create_scratch_key

pytestmark = pytest.mark.asyncio(loop_scope="session")


# Same-team peers can READ each other's keys (see test_key_info) but cannot
# DELETE them — delete is stricter than read.
_SCENARIOS = [
    ("self/proxy_admin", Actor.PROXY_ADMIN, "self", 200),
    ("self/org_admin", Actor.ORG_ADMIN, "self", 401),
    ("self/team_admin", Actor.TEAM_ADMIN, "self", 200),
    ("self/internal_user", Actor.INTERNAL_USER, "self", 200),
    ("self/owner", Actor.OWNER, "self", 200),
    ("self/unrelated_same_org", Actor.UNRELATED_SAME_ORG, "self", 200),
    ("self/cross_org_user", Actor.CROSS_ORG_USER, "self", 200),
    ("self/service_account", Actor.SERVICE_ACCOUNT, "self", 200),
    ("owner_target/proxy_admin", Actor.PROXY_ADMIN, "owner", 200),
    ("owner_target/org_admin", Actor.ORG_ADMIN, "owner", 401),
    ("owner_target/team_admin", Actor.TEAM_ADMIN, "owner", 200),
    ("owner_target/internal_user", Actor.INTERNAL_USER, "owner", 403),
    ("owner_target/unrelated_same_org", Actor.UNRELATED_SAME_ORG, "owner", 403),
    ("owner_target/cross_org_user", Actor.CROSS_ORG_USER, "owner", 403),
    ("owner_target/service_account", Actor.SERVICE_ACCOUNT, "owner", 403),
    ("cross_org_target/proxy_admin", Actor.PROXY_ADMIN, "cross_org", 200),
    ("cross_org_target/org_admin", Actor.ORG_ADMIN, "cross_org", 401),
    ("cross_org_target/team_admin", Actor.TEAM_ADMIN, "cross_org", 403),
    ("cross_org_target/owner", Actor.OWNER, "cross_org", 403),
    ("cross_org_target/cross_org_user", Actor.CROSS_ORG_USER, "cross_org", 200),
    ("cross_org_target/service_account", Actor.SERVICE_ACCOUNT, "cross_org", 403),
]


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
        "/key/delete",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={"keys": [target_cleartext]},
    )
    assert (
        resp.status_code == expected_status
    ), f"{actor.value} {target_shape}: {resp.status_code} {resp.text}"

    row = await prisma.db.litellm_verificationtoken.find_unique(
        where={"token": target_hashed}
    )
    auth_check = await proxy_client.get(
        "/key/info", headers={"Authorization": f"Bearer {target_cleartext}"}
    )

    if expected_status == 200:
        # Hard- or soft-delete both produce a 401 on subsequent auth.
        assert auth_check.status_code == 401
    else:
        assert row is not None, f"{actor.value}: denied but row vanished"
        assert auth_check.status_code == 200


async def test_key_delete_missing_key_is_404(proxy_client, world):
    """Deleting a key absent from the DB is a 404 — not 401/403."""
    resp = await proxy_client.post(
        "/key/delete",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"keys": ["sk-" + uuid.uuid4().hex]},
    )
    assert resp.status_code == 404, resp.text
