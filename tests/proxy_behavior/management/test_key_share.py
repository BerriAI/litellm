import os

import pytest

from .actors import Actor

pytestmark = pytest.mark.asyncio(loop_scope="session")


# POST /key/share is admin-only (proxy admin, the key's team admin, or that
# team's org admin) and needs PASSWORD_LINK_API_KEY set on the proxy. The
# behavior suite runs with that env var unset, so an authorized caller on an
# existing key stops at the config gate (400) rather than hitting password.link
# over the network. These pin the auth, authz, not-found, and config gates.


@pytest.fixture(autouse=True)
def _unconfigured_password_link(monkeypatch):
    monkeypatch.delenv("PASSWORD_LINK_API_KEY", raising=False)


async def test_key_share_requires_auth(proxy_client, world):
    resp = await proxy_client.post(
        "/key/share",
        json={"key": world.keys[Actor.OWNER].cleartext},
    )
    assert resp.status_code == 401, resp.text


async def test_key_share_rejects_malformed_key(proxy_client, world):
    resp = await proxy_client.post(
        "/key/share",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"key": "not a valid key"},
    )
    assert resp.status_code == 400, resp.text
    assert "invalid key format" in resp.text.lower()


async def test_key_share_unknown_key_is_not_found(proxy_client, world):
    resp = await proxy_client.post(
        "/key/share",
        headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
        json={"key": "sk-does-not-exist-000000000000"},
    )
    assert resp.status_code == 404, resp.text


async def test_key_share_denies_non_admin(proxy_client, world):
    resp = await proxy_client.post(
        "/key/share",
        headers={"Authorization": f"Bearer {world.keys[Actor.INTERNAL_USER].cleartext}"},
        json={"key": world.keys[Actor.OWNER].cleartext},
    )
    assert resp.status_code == 403, resp.text


@pytest.mark.parametrize("actor", [Actor.PROXY_ADMIN, Actor.TEAM_ADMIN])
async def test_key_share_admin_reaches_config_gate(actor, proxy_client, world):
    assert "PASSWORD_LINK_API_KEY" not in os.environ
    resp = await proxy_client.post(
        "/key/share",
        headers={"Authorization": f"Bearer {world.keys[actor].cleartext}"},
        json={"key": world.keys[Actor.OWNER].cleartext},
    )
    assert resp.status_code == 400, resp.text
    assert "password.link" in resp.text.lower()
