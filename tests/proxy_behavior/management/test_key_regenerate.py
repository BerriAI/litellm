import litellm
import pytest

from litellm.types.proxy.management_endpoints.ui_sso import (
    LiteLLM_UpperboundKeyGenerateParams,
)

from .actors import TEAM_ALPHA, TEAM_BETA, Actor
from .conftest import create_scratch_key

pytestmark = pytest.mark.asyncio(loop_scope="session")


# Most denials route through team_member_permission (401), unlike /key/update
# which goes through user_id-mismatch (403). The matrix surfaces that
# divergence between the two endpoints.
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
    ("owner_target/internal_user", Actor.INTERNAL_USER, "owner", 401),
    ("owner_target/unrelated_same_org", Actor.UNRELATED_SAME_ORG, "owner", 401),
    ("owner_target/cross_org_user", Actor.CROSS_ORG_USER, "owner", 401),
    ("owner_target/service_account", Actor.SERVICE_ACCOUNT, "owner", 401),
    ("cross_org_target/proxy_admin", Actor.PROXY_ADMIN, "cross_org", 200),
    ("cross_org_target/org_admin", Actor.ORG_ADMIN, "cross_org", 401),
    ("cross_org_target/team_admin", Actor.TEAM_ADMIN, "cross_org", 401),
    ("cross_org_target/owner", Actor.OWNER, "cross_org", 401),
    ("cross_org_target/cross_org_user", Actor.CROSS_ORG_USER, "cross_org", 401),
    ("cross_org_target/service_account", Actor.SERVICE_ACCOUNT, "cross_org", 401),
]


async def _info(proxy_client, cleartext: str):
    return await proxy_client.get(
        "/key/info", headers={"Authorization": f"Bearer {cleartext}"}
    )


@pytest.mark.parametrize(
    "actor,target_shape,expected_status",
    [(a, t, s) for (_id, a, t, s) in _SCENARIOS],
    ids=[s[0] for s in _SCENARIOS],
)
async def test_key_regenerate_authz_matrix(
    actor: Actor,
    target_shape: str,
    expected_status: int,
    proxy_client,
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

    resp = await proxy_client.post(
        "/key/regenerate",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={"key": target_cleartext},
    )
    assert (
        resp.status_code == expected_status
    ), f"{actor.value} {target_shape}: {resp.status_code} {resp.text}"

    if expected_status == 200:
        new_cleartext = resp.json()["key"]
        assert new_cleartext.startswith("sk-") and new_cleartext != target_cleartext
        assert (await _info(proxy_client, target_cleartext)).status_code == 401
        assert (await _info(proxy_client, new_cleartext)).status_code == 200
    else:
        # Denied: rotation must not have leaked — old cleartext still works.
        assert (await _info(proxy_client, target_cleartext)).status_code == 200


async def test_key_path_regenerate_smoke(proxy_client, scratch, world):
    """Pins that POST /key/{key:path}/regenerate shares the same handler."""
    caller = world.keys[Actor.PROXY_ADMIN]
    target_cleartext = await create_scratch_key(
        proxy_client, caller.cleartext, scratch.prefix, user_id=caller.user_id
    )

    resp = await proxy_client.post(
        f"/key/{target_cleartext}/regenerate",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={},
    )
    assert resp.status_code == 200, resp.text
    new_cleartext = resp.json()["key"]
    assert new_cleartext.startswith("sk-") and new_cleartext != target_cleartext
    assert (await _info(proxy_client, target_cleartext)).status_code == 401
    assert (await _info(proxy_client, new_cleartext)).status_code == 200


async def test_key_regenerate_enforces_upperbound_key_params(
    proxy_client, scratch, world, monkeypatch
):
    """Regenerate runs _enforce_upperbound_key_params: a max_budget above
    litellm.upperbound_key_generate_params is rejected 400, a value within the
    bound is accepted. Pins #26340 (db8ef44323) — regenerate previously
    bypassed the upperbound. upperbound_key_generate_params is module-level
    litellm.* state, so monkeypatch save/restores it."""
    admin = world.keys[Actor.PROXY_ADMIN]
    over_key = await create_scratch_key(
        proxy_client,
        admin.cleartext,
        scratch.prefix,
        user_id=admin.user_id,
        key_alias=f"{scratch.prefix}-over",
    )
    within_key = await create_scratch_key(
        proxy_client,
        admin.cleartext,
        scratch.prefix,
        user_id=admin.user_id,
        key_alias=f"{scratch.prefix}-within",
    )
    monkeypatch.setattr(
        litellm,
        "upperbound_key_generate_params",
        LiteLLM_UpperboundKeyGenerateParams(max_budget=100.0),
    )
    headers = {"Authorization": f"Bearer {admin.cleartext}"}

    over = await proxy_client.post(
        "/key/regenerate", headers=headers, json={"key": over_key, "max_budget": 500.0}
    )
    assert over.status_code == 400, over.text

    within = await proxy_client.post(
        "/key/regenerate",
        headers=headers,
        json={"key": within_key, "max_budget": 50.0},
    )
    assert within.status_code == 200, within.text
