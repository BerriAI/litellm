import base64
import json
import os

import httpx
import pytest
import respx

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


@pytest.fixture
def force_httpx_transport(monkeypatch):
    """Make the proxy's outbound HTTP client use httpx's transport so respx can
    intercept the password.link call. litellm defaults to an aiohttp transport
    that respx (an httpx tool) cannot see. Reset the cached client too so a fresh
    httpx-backed one is built; monkeypatch restores both after the test."""
    import litellm
    from litellm.caching.llm_caching_handler import LLMClientCache

    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())


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


async def test_key_share_returns_link_and_never_uploads_plaintext(
    proxy_client, world, monkeypatch, force_httpx_transport
):
    monkeypatch.setenv("PASSWORD_LINK_API_KEY", "private-test-key")
    cleartext = world.keys[Actor.OWNER].cleartext

    with respx.mock(assert_all_called=True) as mock:
        route = mock.post("https://password.link/api/secrets").mock(
            return_value=httpx.Response(201, json={"data": {"id": "abc123"}})
        )
        mock.route().pass_through()
        resp = await proxy_client.post(
            "/key/share",
            headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
            json={"key": cleartext, "expiration_hours": 12, "max_views": 1},
        )

    assert resp.status_code == 200, resp.text
    assert resp.json()["share_link"].startswith("https://password.link/?abc123#")

    request_body = json.loads(route.calls.last.request.content)
    assert route.calls.last.request.headers["authorization"] == "ApiKey private-test-key"
    assert request_body["expiration"] == 12
    assert cleartext not in json.dumps(request_body)
    assert cleartext not in base64.b64decode(request_body["ciphertext"]).decode("utf-8")


async def test_key_share_maps_upstream_failure_to_502(proxy_client, world, monkeypatch, force_httpx_transport):
    monkeypatch.setenv("PASSWORD_LINK_API_KEY", "private-test-key")

    with respx.mock(assert_all_called=True) as mock:
        mock.post("https://password.link/api/secrets").mock(return_value=httpx.Response(402, text="quota exceeded"))
        mock.route().pass_through()
        resp = await proxy_client.post(
            "/key/share",
            headers={"Authorization": f"Bearer {world.keys[Actor.PROXY_ADMIN].cleartext}"},
            json={"key": world.keys[Actor.OWNER].cleartext},
        )

    assert resp.status_code == 502, resp.text
    assert "402" in resp.text
