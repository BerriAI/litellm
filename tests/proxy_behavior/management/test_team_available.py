import pytest

from .actors import Actor

pytestmark = pytest.mark.asyncio(loop_scope="session")


# GET /team/available lists teams from
# litellm.default_internal_user_params["available_teams"]. The behavior world
# configures no available_teams, so the handler returns [] for every actor
# before it even reads the caller — this is the route-coverage + default-path
# pin. /team/available is an info route, so every authenticated actor reaches
# the handler.
@pytest.mark.parametrize("actor", list(Actor), ids=[a.value for a in Actor])
async def test_team_available_default_is_empty(actor: Actor, proxy_client, world):
    resp = await proxy_client.get(
        "/team/available",
        headers={"Authorization": f"Bearer {world.keys[actor].cleartext}"},
    )
    assert resp.status_code == 200, f"{actor.value}: {resp.status_code} {resp.text}"
    assert resp.json() == []
