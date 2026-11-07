import json
from typing import Final

import pytest

from tests.integration._support.actors import TEAM_ALPHA, TEAM_BETA, Actor
from tests.integration._support.actors import create_scratch_key, owned_settings_client
from tests.integration._support.client import Gateway, object_value
from tests.integration._support.wire import Reply, Request, wire_server

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


async def _info(gateway_client, cleartext: str):
    return await gateway_client.get(
        "/key/info", headers={"Authorization": f"Bearer {cleartext}"}
    )


async def test_master_key_regeneration_is_rejected_without_rotating_model(gateway: Gateway):
    master_key_regeneration_error: Final = (
        "Rotating the master key through /key/regenerate is no longer supported. "
        "Set the new value in LITELLM_MASTER_KEY or general_settings.master_key and restart every proxy instance. "
        "If LITELLM_SALT_KEY is not set, first set it to your current master key. "
        "See https://docs.litellm.ai/docs/proxy/master_key_rotations"
    )

    def upstream(_: Request) -> Reply:
        return Reply(
            body=json.dumps(
                {
                    "id": "master-key-regeneration-response",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "available"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            ).encode()
        )

    with gateway.scenario() as scenario, wire_server(upstream) as provider:
        model: Final = scenario.model(api_base=f"{provider.url}/v1")
        response: Final = gateway.request(
            "POST",
            "/key/regenerate",
            {"key": gateway.key, "new_master_key": "sk-new-master-key"},
        )
        assert response.status_code == 400, response.text
        assert response.json()["error"]["message"] == str({"error": master_key_regeneration_error})

        model_info: Final = gateway.request("GET", "/model/info")
        assert model_info.status_code == 200, model_info.text
        listed: Final = model_info.json()["data"]
        assert isinstance(listed, list)
        assert any(object_value(entry)["model_name"] == model for entry in listed)
        served: Final = gateway.chat(model, text="master key regeneration")
        assert object_value(served["choices"][0])["message"] == {
            "role": "assistant",
            "content": "available",
        }


@pytest.mark.parametrize(
    "actor,target_shape,expected_status",
    [(a, t, s) for (_id, a, t, s) in _SCENARIOS],
    ids=[s[0] for s in _SCENARIOS],
)
async def test_key_regenerate_authz_matrix(
    actor: Actor,
    target_shape: str,
    expected_status: int,
    gateway_client,
    resources,
    actors,
):
    caller = actors.keys[actor]
    seeder = actors.keys[Actor.PROXY_ADMIN].cleartext

    if target_shape == "self":
        target_cleartext = await create_scratch_key(
            gateway_client, seeder, resources.prefix, user_id=caller.user_id
        )
    elif target_shape == "owner":
        target_cleartext = await create_scratch_key(
            gateway_client,
            seeder,
            resources.prefix,
            user_id=actors.keys[Actor.OWNER].user_id,
            team_id=actors.team_alpha_id,
        )
    elif target_shape == "cross_org":
        target_cleartext = await create_scratch_key(
            gateway_client,
            seeder,
            resources.prefix,
            user_id=actors.keys[Actor.CROSS_ORG_USER].user_id,
            team_id=actors.team_beta_id,
        )
    else:
        pytest.fail(f"unknown target_shape={target_shape}")

    resp = await gateway_client.post(
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
        assert (await _info(gateway_client, target_cleartext)).status_code == 401
        assert (await _info(gateway_client, new_cleartext)).status_code == 200
    else:
        # Denied: rotation must not have leaked — old cleartext still works.
        assert (await _info(gateway_client, target_cleartext)).status_code == 200


async def test_key_path_regenerate_smoke(gateway_client, resources, actors):
    """Pins that POST /key/{key:path}/regenerate shares the same handler."""
    caller = actors.keys[Actor.PROXY_ADMIN]
    target_cleartext = await create_scratch_key(
        gateway_client, caller.cleartext, resources.prefix, user_id=caller.user_id
    )

    resp = await gateway_client.post(
        f"/key/{target_cleartext}/regenerate",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
        json={},
    )
    assert resp.status_code == 200, resp.text
    new_cleartext = resp.json()["key"]
    assert new_cleartext.startswith("sk-") and new_cleartext != target_cleartext
    assert (await _info(gateway_client, target_cleartext)).status_code == 401
    assert (await _info(gateway_client, new_cleartext)).status_code == 200


async def test_key_regenerate_enforces_upperbound_key_params(
    gateway_client, resources, actors, tmp_path
):
    admin = actors.keys[Actor.PROXY_ADMIN]
    over_key = await create_scratch_key(
        gateway_client,
        admin.cleartext,
        resources.prefix,
        user_id=admin.user_id,
        key_alias=f"{resources.prefix}-over",
    )
    within_key = await create_scratch_key(
        gateway_client,
        admin.cleartext,
        resources.prefix,
        user_id=admin.user_id,
        key_alias=f"{resources.prefix}-within",
    )
    headers = {"Authorization": f"Bearer {admin.cleartext}"}

    with owned_settings_client(
        gateway_client, tmp_path, {"upperbound_key_generate_params": {"max_budget": 100.0}}
    ) as client:
        over = await client.post(
            "/key/regenerate", headers=headers, json={"key": over_key, "max_budget": 500.0}
        )
        within = await client.post(
            "/key/regenerate",
            headers=headers,
            json={"key": within_key, "max_budget": 50.0},
        )
    assert over.status_code == 400, over.text
    assert within.status_code == 200, within.text
