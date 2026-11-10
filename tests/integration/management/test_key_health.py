import pytest

from tests.integration._support.actors import Actor

pytestmark = pytest.mark.asyncio(loop_scope="session")


# POST /key/health has no role gate — it reflects the caller's OWN key logging
# metadata. The actors keys carry no "logging" metadata, so every authenticated
# actor gets 200 with key="healthy". This pins auth-required + route coverage.
@pytest.mark.parametrize("actor", list(Actor), ids=[a.value for a in Actor])
async def test_key_health_each_actor_is_healthy(actor: Actor, gateway_client, actors):
    caller = actors.keys[actor]
    resp = await gateway_client.post(
        "/key/health",
        headers={"Authorization": f"Bearer {caller.cleartext}"},
    )
    assert resp.status_code == 200, f"{actor.value}: {resp.status_code} {resp.text}"
    assert resp.json()["key"] == "healthy"


async def test_key_health_requires_auth(gateway_client):
    resp = await gateway_client.post("/key/health", headers={"Authorization": "Bearer invalid"})
    assert resp.status_code == 401, resp.text
