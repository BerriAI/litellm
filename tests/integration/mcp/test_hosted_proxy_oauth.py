import hashlib
import json
import os
import re
import secrets
from base64 import urlsafe_b64encode
from contextlib import ExitStack
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi import HTTPException
from fastapi import Request as FastAPIRequest
from integration._support.client import Gateway
from integration._support.oauth_server import oauth_server
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, wire_server

from litellm.caching.redis_cache import RedisCache
from litellm.proxy._experimental.mcp_server.hosted_proxy_auth import HostedFailure, HostedGrant, RedisHostedGrantStore
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

CALLBACK: Final = "https://admin.example/oauth/callback"


async def hosted_custom_policy(request: FastAPIRequest, api_key: str) -> UserAPIKeyAuth:
    if api_key == os.environ["LITELLM_MASTER_KEY"]:
        return UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    raise HTTPException(status_code=403, detail="custom policy denied this request")


@pytest.mark.asyncio
@pytest.mark.parametrize("spent_token", ["first-refresh", "first-access"])
async def test_shared_redis_atomically_consumes_codes_and_rejects_stale_refresh_writes(spent_token: str) -> None:
    cache: Final = RedisCache(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]))
    store: Final = RedisHostedGrantStore(cache, "sk-" + secrets.token_urlsafe(32))
    grant: Final = HostedGrant(
        grant_id=secrets.token_urlsafe(24),
        user_id="user",
        team_id=None,
        client_id="client",
        redirect_uri=CALLBACK,
        resource="https://gateway.example",
        scope="proxy:read",
        expires_at=200,
        access_expires_at=100,
        access_hash="first-access",
        refresh_hash="first-refresh",
    )
    rotated: Final = grant.model_copy(update={"access_hash": "second-access", "refresh_hash": "second-refresh"})
    try:
        assert await store.replace(grant, None, 60)
        assert not await store.replace(grant, None, 60)
        assert await store.replace(rotated, grant, 60)
        assert not await store.replace(grant, grant, 60)
        assert await store.read(grant.grant_id) == rotated
        await store.revoke_replayed(grant.grant_id, "unknown-refresh")
        assert await store.read(grant.grant_id) == rotated
        await store.revoke_replayed(grant.grant_id, spent_token)
        assert isinstance(await store.read(grant.grant_id), HostedFailure)
        assert not await store.replace(grant, None, 60)
        assert not await store.replace(grant, rotated, 60)
    finally:
        await store.delete(grant.grant_id)
        await cache.disconnect()


@pytest.mark.timeout(300)
@pytest.mark.parametrize("exclusive_auth", ["custom_auth", "enable_oauth2_auth", "enable_oauth2_proxy_auth"])
@pytest.mark.parametrize("scope", ["proxy:read", "proxy:admin"])
def test_hosted_sso_return_permissions_refresh_and_revocation(
    gateway: Gateway, tmp_path: Path, exclusive_auth: str, scope: str
) -> None:
    with oauth_server(scopes=("openid", "email", "profile")) as idp:
        email: Final = f"hosted-{secrets.token_hex(8)}@example.com"

        def userinfo(request: Request) -> Reply:
            token: Final = request.headers.get("authorization", "").removeprefix("Bearer ")
            assert idp.is_live(token)
            return Reply(body=json.dumps({"sub": email, "email": email, "role": "proxy_admin"}).encode())

        with (
            wire_server(userinfo) as profile,
            owned_proxy_process(
                gateway,
                tmp_path,
                MappingProxyType(
                    {
                        "LITELLM_PROXY_API_OAUTH_REDIRECT_URIS": CALLBACK,
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
                ),
                remove_environment=("PROXY_BASE_URL", "GOOGLE_CLIENT_ID", "MICROSOFT_CLIENT_ID"),
            ) as owned,
            owned.gateway.scenario() as scenario,
            ExitStack() as peers,
        ):
            user_id: Final = scenario.user(user_id=email, user_email=email, user_role="proxy_admin")
            base: Final = str(owned.gateway.client.base_url).rstrip("/")
            policy_config: Final = tmp_path / "custom-policy.yaml"
            policy: Final = (
                "custom_auth: integration.mcp.test_hosted_proxy_oauth.hosted_custom_policy"
                if exclusive_auth == "custom_auth"
                else f"{exclusive_auth}: true"
            )
            policy_config.write_text(
                (Path(__file__).resolve().parents[1] / "proxy_config.yaml")
                .read_text()
                .replace(
                    "general_settings:\n",
                    f"general_settings:\n  {policy}\n",
                )
            )
            restricted: Final = peers.enter_context(owned_proxy_process(
                gateway,
                tmp_path,
                {"PROXY_BASE_URL": base, "LITELLM_PROXY_API_OAUTH_REDIRECT_URIS": CALLBACK,
                 "LITELLM_PROXY_API_OAUTH_ADMIN_REDIRECT_URIS": CALLBACK},
                config=policy_config,
            ))
            verifier: Final = secrets.token_urlsafe(32)
            challenge: Final = urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
            registered: Final = owned.gateway.client.post("/register", json={"redirect_uris": [CALLBACK]})
            assert registered.status_code == 201, registered.text
            client_id: Final = registered.json()["client_id"]
            assert registered.json()["grant_types"] == ["authorization_code", "refresh_token"]
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
                        "scope": scope,
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
                assert consent.status_code == 200, consent.text
                assert ("administrator permissions" if scope == "proxy:admin" else "read-only model listings") in consent.text
                assert "personal credential" not in consent.text
                handle: Final = re.search(r'name="flow" value="([^"]+)"', consent.text)
                assert handle is not None, consent.text
                approved: Final = browser.post(
                    "/authorize/complete", data={"flow": handle.group(1), "decision": "approve"}
                )
                assert approved.status_code == 303, approved.text
                location: Final = approved.headers["location"]
                assert location.startswith(CALLBACK + "?"), location
                params: Final = parse_qs(urlsplit(location).query)
                assert params["state"] == ["hosted-client-state"]
                assert set(params) == {"code", "state"}
                code: Final = params["code"][0]

            exchange: Final = {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": CALLBACK,
                "client_id": client_id,
                "code_verifier": verifier,
                "resource": base,
            }
            stolen: Final = owned.gateway.client.post("/token", data={**exchange, "code_verifier": "x" * 43})
            assert stolen.status_code == 400 and stolen.json()["error"] == "invalid_grant", stolen.text
            blocked_issue: Final = restricted.gateway.client.post("/token", data=exchange)
            assert blocked_issue.status_code == 400, blocked_issue.text
            assert "custom or external auth is configured" in blocked_issue.json()["error_description"]
            redeemed: Final = owned.gateway.client.post("/token", data=exchange)
            assert redeemed.status_code == 200, redeemed.text
            tokens: Final = redeemed.json()
            assert tokens["scope"] == scope and tokens["user_id"] == user_id
            headers: Final = {"Authorization": "Bearer " + tokens["access_token"]}
            other_gateway: Final = Gateway(gateway.client, "sk-other-hosted-test-master", gateway.upstream_url)
            with owned_proxy_process(
                other_gateway,
                tmp_path,
                {"LITELLM_PROXY_API_OAUTH_REDIRECT_URIS": CALLBACK},
                remove_environment=("PROXY_BASE_URL",),
            ) as other:
                cross_gateway: Final = other.gateway.client.get(
                    "/v1/models",
                    headers={**headers, "Host": urlsplit(base).netloc},
                )
                assert cross_gateway.status_code == 401, cross_gateway.text
            models: Final = owned.gateway.client.get("/v1/models", headers=headers)
            assert models.status_code == 200, models.text
            for method, route in (() if scope == "proxy:admin" else (
                ("POST", "/key/generate"),
                ("GET", "/key/info"),
                ("POST", "/v1/chat/completions"),
            )):
                denied: Final = owned.gateway.client.request(method, route, headers=headers, json={})
                assert denied.status_code == 403, denied.text
            if scope == "proxy:admin":
                _exercise_admin(owned.gateway, headers, user_id)
            demoted: Final = owned.gateway.request(
                "POST", "/user/update", {"user_id": user_id, "user_role": "internal_user"}
            )
            assert demoted.status_code == 200, demoted.text
            report: Final = owned.gateway.client.get("/global/spend", headers=headers)
            assert report.status_code == 403, report.text
            if scope == "proxy:admin":
                denied_refresh: Final = owned.gateway.client.post("/token", data={
                    "grant_type": "refresh_token", "client_id": client_id,
                    "refresh_token": tokens["refresh_token"], "resource": base,
                })
                assert denied_refresh.status_code == 400, denied_refresh.text
                promoted: Final = owned.gateway.request(
                    "POST", "/user/update", {"user_id": user_id, "user_role": "proxy_admin"}
                )
                assert promoted.status_code == 200, promoted.text
            rotated: Final = owned.gateway.client.post(
                "/token",
                data={
                    "grant_type": "refresh_token",
                    "client_id": client_id,
                    "refresh_token": tokens["refresh_token"],
                    "resource": base,
                },
            )
            assert rotated.status_code == 200, rotated.text
            current: Final = rotated.json()
            in_flight: Final = owned.gateway.client.get("/v1/models", headers=headers)
            assert in_flight.status_code == 200, in_flight.text
            blocked_access: Final = restricted.gateway.client.get(
                "/v1/models", headers={"Authorization": "Bearer " + current["access_token"]}
            )
            assert blocked_access.status_code == 401, blocked_access.text
            assert "custom or external auth is configured" in blocked_access.text
            blocked_refresh: Final = restricted.gateway.client.post(
                "/token",
                data={
                    "grant_type": "refresh_token",
                    "client_id": client_id,
                    "refresh_token": current["refresh_token"],
                    "resource": base,
                },
            )
            assert blocked_refresh.status_code == 400, blocked_refresh.text
            assert "custom or external auth is configured" in blocked_refresh.json()["error_description"]
            revoked: Final = restricted.gateway.client.post(
                "/revoke", data={"token": current["refresh_token"], "client_id": client_id}
            )
            assert revoked.status_code == 200, revoked.text
            after: Final = owned.gateway.client.get(
                "/v1/models", headers={"Authorization": "Bearer " + current["access_token"]}
            )
            assert after.status_code == 401, after.text
            replayed: Final = owned.gateway.client.post("/token", data=exchange)
            assert replayed.status_code == 400, replayed.text
            log: Final = owned.log.read_text()
            assert tokens["access_token"] not in log
            assert tokens["refresh_token"] not in log


def _exercise_admin(gateway: Gateway, headers: dict[str, str], user_id: str) -> None:
    user: Final = gateway.client.get("/user/info", headers=headers)
    assert user.status_code == 200 and user.json()["user_id"] == user_id, user.text
    created: Final = gateway.client.post("/key/generate", headers=headers, json={"user_id": user_id})
    assert created.status_code == 200, created.text
    key: Final = created.json()["key"]
    try:
        fetched: Final = gateway.client.get("/key/info", headers=headers, params={"key": key})
        assert fetched.status_code == 200 and fetched.json()["info"]["user_id"] == user_id, fetched.text
    finally:
        deleted: Final = gateway.client.post("/key/delete", headers=headers, json={"keys": [key]})
        assert deleted.status_code == 200, deleted.text

    def completion(request: Request) -> Reply:
        assert request.target == "/v1/chat/completions"
        assert json.loads(request.body)["messages"] == [{"role": "user", "content": "hosted grant"}]
        return Reply(body=json.dumps({
            "id": "hosted-completion", "object": "chat.completion", "created": 1, "model": "test",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }).encode())

    with wire_server(completion) as upstream, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=upstream.url + "/v1")
        response: Final = gateway.client.post("/v1/chat/completions", headers=headers, json={
            "model": model, "messages": [{"role": "user", "content": "hosted grant"}],
        })
        assert response.status_code == 200 and response.json()["choices"][0]["message"]["content"] == "ok", response.text
