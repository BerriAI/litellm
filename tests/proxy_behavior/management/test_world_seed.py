import pytest

from .actors import Actor

pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest.mark.parametrize("actor", list(Actor), ids=[a.value for a in Actor])
async def test_each_actor_can_self_info(actor, proxy_client, world):
    seeded = world.keys[actor]
    resp = await proxy_client.get(
        "/key/info",
        headers={"Authorization": f"Bearer {seeded.cleartext}"},
    )
    assert resp.status_code == 200, f"{actor.value}: {resp.text}"
    body = resp.json()
    assert body.get("key") == seeded.hashed
    assert body["info"].get("user_id") == seeded.user_id


async def test_proxy_admin_actor_can_create_keys_for_others(proxy_client, world):
    seeder = world.keys[Actor.PROXY_ADMIN]
    target_user_id = world.keys[Actor.OWNER].user_id

    resp = await proxy_client.post(
        "/key/generate",
        headers={"Authorization": f"Bearer {seeder.cleartext}"},
        json={"key_alias": "smoke-proxy-admin-bypass", "user_id": target_user_id},
    )
    assert resp.status_code == 200, resp.text
