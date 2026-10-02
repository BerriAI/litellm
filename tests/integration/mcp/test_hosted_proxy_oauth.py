import asyncio
import hashlib
import json
import os
import re
import secrets
from base64 import urlsafe_b64encode
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from _fake_openai_endpoint_server import _chat_completion_body
from integration._support.client import Gateway
from integration._support.oauth_server import oauth_server
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, wire_server

from litellm.caching.redis_cache import RedisCache
from litellm.proxy._experimental.mcp_server.hosted_proxy_auth import RedisHostedGrantStore

CALLBACK: Final = "https://admin.example/oauth/callback"


@pytest.mark.asyncio
async def test_shared_redis_serializes_rotation_and_revocation_without_reviving_grants() -> None:
    cache: Final = RedisCache(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]))
    store: Final = RedisHostedGrantStore(cache, "sk-" + secrets.token_urlsafe(32))
    grant_id, racing_id = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
    try:
        issued: Final = await asyncio.gather(*(store.advance(grant_id, None, "first", 60) for _ in range(2)))
        assert sorted(issued) == [False, True]
        rotated: Final = await asyncio.gather(
            *(store.advance(grant_id, "first", token, 60) for token in ("second", "third"))
        )
        assert sorted(rotated) == [False, True]
        assert await store.read(grant_id) == "revoked"
        assert await store.advance(racing_id, None, "first", 60)
        await asyncio.gather(store.revoke(racing_id), store.advance(racing_id, "first", "second", 60))
        for identity in (grant_id, racing_id):
            assert await store.read(identity) == "revoked"
            assert not await store.advance(identity, None, "resurrected", 60)
            assert not await store.advance(identity, "first", "resurrected", 60)
            assert await store.read(identity) == "revoked"
    finally:
        await cache.delete_cache_keys([store._key(grant_id), store._key(racing_id)])
        await cache.disconnect()


@pytest.mark.timeout(300)
def test_hosted_admin_uses_configured_sso_and_revocable_sessions_across_instances(
    gateway: Gateway, tmp_path: Path
) -> None:
    with oauth_server(scopes=("openid", "email", "profile")) as idp:
        email: Final = f"hosted-{secrets.token_hex(8)}@example.com"

        def userinfo(request: Request) -> Reply:
            assert idp.is_live(request.headers.get("authorization", "").removeprefix("Bearer "))
            return Reply(body=json.dumps({"sub": email, "email": email, "role": "proxy_admin"}).encode())

        with wire_server(userinfo) as profile:
            environment: Final = {
                "LITELLM_PROXY_API_OAUTH_ADMIN_REDIRECT_URIS": CALLBACK,
                "GENERIC_CLIENT_ID": "hosted-sso-test",
                "GENERIC_CLIENT_SECRET": "hosted-sso-test-secret",
                "GENERIC_AUTHORIZATION_ENDPOINT": idp.issuer + "/authorize",
                "GENERIC_TOKEN_ENDPOINT": idp.issuer + "/token",
                "GENERIC_USERINFO_ENDPOINT": profile.url + "/userinfo",
                "GENERIC_CLIENT_USE_PKCE": "true",
                "GENERIC_USER_ID_ATTRIBUTE": "sub",
                "GENERIC_USER_ROLE_ATTRIBUTE": "role",
            }
            with (
                owned_proxy_process(
                    gateway,
                    tmp_path,
                    environment,
                    remove_environment=("PROXY_BASE_URL", "GOOGLE_CLIENT_ID", "MICROSOFT_CLIENT_ID"),
                ) as owned,
                owned.gateway.scenario() as scenario,
            ):
                user_id: Final = scenario.user(user_id=email, user_email=email, user_role="proxy_admin")
                base: Final = str(owned.gateway.client.base_url).rstrip("/")
                verifier: Final = secrets.token_urlsafe(32)
                challenge: Final = urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
                registered: Final = owned.gateway.client.post("/register", json={"redirect_uris": [CALLBACK]})
                assert registered.status_code == 201, registered.text
                client_id: Final = registered.json()["client_id"]
                with httpx.Client(base_url=base, follow_redirects=False, trust_env=False, timeout=15) as browser:
                    started: Final = browser.get(
                        "/authorize",
                        params={
                            "client_id": client_id,
                            "redirect_uri": CALLBACK,
                            "resource": base,
                            "response_type": "code",
                            "code_challenge": challenge,
                            "code_challenge_method": "S256",
                            "state": "hosted-client-state",
                            "scope": "proxy:admin",
                        },
                    )
                    assert started.status_code == 303, started.text
                    assert urlsplit(started.headers["location"]).path == "/sso/key/generate"
                    login: Final = browser.get(started.headers["location"])
                    assert login.status_code in (302, 303, 307), login.text
                    assert login.headers["location"].startswith(idp.issuer + "/authorize")
                    idp_approved: Final = browser.get(login.headers["location"])
                    assert idp_approved.status_code == 302, idp_approved.text
                    returned: Final = browser.get(idp_approved.headers["location"])
                    assert returned.status_code == 303, returned.text
                    assert returned.headers["location"].startswith("/authorize?")
                    consent: Final = browser.get(returned.headers["location"])
                    assert consent.status_code == 200 and "administrator permissions" in consent.text, consent.text
                    handle: Final = re.search(r'name="flow" value="([^"]+)"', consent.text)
                    assert handle is not None, consent.text
                    approved: Final = browser.post(
                        "/authorize/complete", data={"flow": handle.group(1), "decision": "approve"}
                    )
                    assert approved.status_code == 303, approved.text
                    location: Final = approved.headers["location"]
                    assert location.startswith(CALLBACK + "?"), location
                    params: Final = parse_qs(urlsplit(location).query)
                    assert set(params) == {"code", "state"} and params["state"] == ["hosted-client-state"]

                exchange: Final = {
                    "grant_type": "authorization_code",
                    "code": params["code"][0],
                    "redirect_uri": CALLBACK,
                    "client_id": client_id,
                    "code_verifier": verifier,
                    "resource": base,
                }
                with owned_proxy_process(gateway, tmp_path, {**environment, "PROXY_BASE_URL": base}) as peer:
                    stolen: Final = peer.gateway.client.post("/token", data={**exchange, "code_verifier": "x" * 43})
                    assert stolen.status_code == 400 and stolen.json()["error"] == "invalid_grant", stolen.text
                    redeemed: Final = peer.gateway.client.post("/token", data=exchange)
                    assert redeemed.status_code == 200, redeemed.text
                    tokens: Final = redeemed.json()
                    assert tokens["scope"] == "proxy:admin" and tokens["user_id"] == user_id
                    headers: Final = {"Authorization": "Bearer " + tokens["access_token"]}
                    _exercise_admin(peer.gateway, headers, user_id)
                    _assert_active(peer.gateway, tokens["access_token"], True)
                    refresh: Final = {
                        "grant_type": "refresh_token",
                        "client_id": client_id,
                        "refresh_token": tokens["refresh_token"],
                        "resource": base,
                    }
                    owned.gateway.post("/user/update", {"user_id": user_id, "user_role": "internal_user"})
                    demoted: Final = peer.gateway.client.get("/global/spend", headers=headers)
                    assert demoted.status_code == 403, demoted.text
                    _assert_active(peer.gateway, tokens["access_token"], False)
                    denied_refresh: Final = peer.gateway.client.post("/token", data=refresh)
                    assert denied_refresh.status_code == 400, denied_refresh.text
                    owned.gateway.post("/user/update", {"user_id": user_id, "user_role": "proxy_admin"})
                    rotated: Final = owned.gateway.client.post("/token", data=refresh)
                    assert rotated.status_code == 200, rotated.text
                    current: Final = rotated.json()
                    assert current["refresh_token"] != tokens["refresh_token"]
                    _assert_active(peer.gateway, tokens["refresh_token"], False)
                    in_flight: Final = peer.gateway.client.get("/v1/models", headers=headers)
                    assert in_flight.status_code == 200, in_flight.text
                    revoked: Final = peer.gateway.client.post(
                        "/revoke", data={"token": tokens["access_token"], "client_id": client_id}
                    )
                    assert revoked.status_code == 200, revoked.text
                    for access in (tokens["access_token"], current["access_token"]):
                        after: Final = owned.gateway.client.get(
                            "/v1/models", headers={"Authorization": "Bearer " + access}
                        )
                        assert after.status_code == 401, after.text
                    after_refresh: Final = owned.gateway.client.post(
                        "/token", data={**refresh, "refresh_token": current["refresh_token"]}
                    )
                    assert after_refresh.status_code == 400, after_refresh.text
                    replayed: Final = peer.gateway.client.post("/token", data=exchange)
                    assert replayed.status_code == 400, replayed.text
                    for process in (owned, peer):
                        log: Final = process.log.read_text()
                        assert tokens["access_token"] not in log and tokens["refresh_token"] not in log


def _exercise_admin(gateway: Gateway, headers: dict[str, str], user_id: str) -> None:
    user: Final = gateway.client.get("/user/info", headers=headers)
    assert user.status_code == 200 and user.json()["user_id"] == user_id, user.text
    minted: Final = gateway.client.post("/key/generate", headers=headers, json={"user_id": user_id})
    assert minted.status_code == 403, minted.text

    expected: Final = _chat_completion_body("hosted-model")
    with wire_server(lambda _: Reply(body=json.dumps(expected).encode())) as upstream, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=upstream.url + "/v1")
        response: Final = gateway.chat(model, key=headers["Authorization"].removeprefix("Bearer "), text="hosted grant")
        assert response["choices"][0]["message"]["content"] == expected["choices"][0]["message"]["content"]
        sent: Final = upstream.received.get_nowait()
        assert sent.target == "/v1/chat/completions"
        assert json.loads(sent.body)["messages"] == [{"role": "user", "content": "hosted grant"}]


def _assert_active(gateway: Gateway, token: str, active: bool) -> None:
    response: Final = gateway.client.post(
        "/introspect", data={"token": token}, headers={"Authorization": "Bearer " + gateway.key}
    )
    assert response.status_code == 200 and response.json()["active"] is active, response.text
