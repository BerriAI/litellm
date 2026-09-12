"""GET /team/metadata_schema — the behavior world declares no
``general_settings.team_metadata_schema``, so the route is an info route that
returns an empty field list to every authenticated actor and 401s without a key.
"""

import pytest

from .actors import Actor

pytestmark = pytest.mark.asyncio(loop_scope="session")


@pytest.mark.parametrize("actor", list(Actor), ids=[a.value for a in Actor])
async def test_team_metadata_schema_default_is_empty(actor: Actor, proxy_client, world):
    resp = await proxy_client.get(
        "/team/metadata_schema",
        headers={"Authorization": f"Bearer {world.keys[actor].cleartext}"},
    )
    assert resp.status_code == 200, f"{actor.value}: {resp.status_code} {resp.text}"
    assert resp.json() == {"fields": []}


async def test_team_metadata_schema_requires_auth(proxy_client, world):
    resp = await proxy_client.get("/team/metadata_schema")
    assert resp.status_code == 401, resp.text
