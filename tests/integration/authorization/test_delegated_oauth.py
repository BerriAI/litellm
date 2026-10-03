import base64
import hashlib
import re
import secrets
import time
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs, urlsplit

import httpx
import jwt
import yaml
from integration._support.client import JSON_OBJECT, Gateway, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy

RESOURCE: Final = "https://gateway.integration.example"
CALLBACK: Final = "https://app.integration.example/callback"


def _authorize(gateway: Gateway, user: str) -> dict[str, str]:
    registration: Final = gateway.client.post("/register", json={"redirect_uris": [CALLBACK]})
    assert registration.status_code == 201, registration.text
    client: Final = string_value(JSON_OBJECT.validate_json(registration.content)["client_id"])
    verifier: Final = secrets.token_urlsafe(48)
    challenge: Final = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    cookies: Final = {
        "token": jwt.encode(
            {"user_id": user, "login_method": "username_password", "exp": int(time.time()) + 600},
            gateway.key,
            algorithm="HS256",
        )
    }
    consent: Final = gateway.client.get(
        "/authorize",
        params={
            "client_id": client,
            "redirect_uri": CALLBACK,
            "response_type": "code",
            "state": user,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "resource": RESOURCE,
            "scope": "proxy:admin",
        },
        cookies=cookies,
    )
    assert consent.status_code == 200, consent.text
    flow: Final = re.search(r'name="flow" value="([^"]+)"', consent.text)
    assert flow is not None, consent.text
    approved: Final = gateway.client.post(
        "/authorize/complete",
        data={"flow": flow[1], "decision": "approve"},
        cookies={**cookies, **dict(consent.cookies)},
    )
    assert approved.status_code == 303, approved.text
    callback: Final = parse_qs(urlsplit(approved.headers["location"]).query)
    assert callback["state"] == [user]
    return {
        "grant_type": "authorization_code",
        "client_id": client,
        "code": callback["code"][0],
        "redirect_uri": CALLBACK,
        "code_verifier": verifier,
        "resource": RESOURCE,
    }


def _refresh(gateway: Gateway, client: str, token: str) -> httpx.Response:
    return gateway.client.post(
        "/token",
        data={
            "grant_type": "refresh_token",
            "client_id": client,
            "refresh_token": token,
            "resource": RESOURCE,
        },
    )


def test_delegated_grants_share_rotation_revocation_and_live_role_checks(gateway: Gateway, tmp_path: Path) -> None:
    overrides: Final = {"PROXY_BASE_URL": RESOURCE, "LITELLM_OAUTH_ADMIN_REDIRECT_URIS": CALLBACK}
    with (
        owned_proxy(gateway, tmp_path, overrides) as first,
        owned_proxy(gateway, tmp_path, overrides) as second,
        first.scenario() as scenario,
    ):
        admin: Final = scenario.user(user_role="proxy_admin")
        for outcome in ("replay", "revoke", "demote"):
            grant: Final = _authorize(first, admin)
            client: Final = grant["client_id"]
            if outcome == "replay":
                first.post("/user/update", {"user_id": admin, "user_role": "internal_user"})
                assert second.client.post("/token", data=grant).status_code == 400
                first.post("/user/update", {"user_id": admin, "user_role": "proxy_admin"})
            issued: Final = first.client.post("/token", data=grant)
            assert issued.status_code == 200, issued.text
            original: Final = JSON_OBJECT.validate_json(issued.content)
            access: Final = string_value(original["access_token"])
            refresh: Final = string_value(original["refresh_token"])
            created: Final = first.post("/key/generate", {"user_id": admin}, key=access)
            key: Final = string_value(created["key"])
            scenario.cleanups.callback(scenario.delete_key, key)
            read: Final = second.request("GET", "/key/info", key=access, params={"key": key})
            assert read.status_code == 200, read.text
            assert object_value(JSON_OBJECT.validate_json(read.content)["info"])["user_id"] == admin
            rotated: Final = _refresh(second, client, refresh)
            assert rotated.status_code == 200, rotated.text
            pair: Final = JSON_OBJECT.validate_json(rotated.content)
            next_access: Final = string_value(pair["access_token"])
            next_refresh: Final = string_value(pair["refresh_token"])
            assert next_refresh != refresh
            assert second.client.post("/token", data=grant).status_code == 400
            warmed: Final = first.request("GET", "/user/info", key=next_access, params={"user_id": admin})
            assert warmed.status_code == 200, warmed.text
            if outcome == "replay":
                assert _refresh(first, client, refresh).status_code == 400
            elif outcome == "revoke":
                revoked: Final = first.client.post("/revoke", data={"client_id": client, "token": next_refresh})
                assert revoked.status_code == 200, revoked.text
            else:
                first.post("/user/update", {"user_id": admin, "user_role": "internal_user"})
            for worker in (first, second):
                assert worker.client.post("/token", data=grant).status_code == 400
                for bearer in (access, next_access):
                    denied: Final = worker.request("GET", "/user/info", key=bearer, params={"user_id": admin})
                    assert denied.status_code in (401, 403), denied.text
                renewal: Final = _refresh(worker, client, next_refresh)
                assert renewal.status_code == 400, renewal.text


def test_delegated_inference_keeps_user_models_budgets_and_attribution(gateway: Gateway, tmp_path: Path) -> None:
    config: Final = JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    path: Final = tmp_path / "delegated-model-policy.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **config,
                "general_settings": {
                    **object_value(config["general_settings"]),
                    "enable_jwt_auth": True,
                    "pass_through_endpoints": [
                        {
                            "path": "/v1/chat/completions",
                            "target": gateway.upstream_url,
                            "methods": ["GET"],
                            "auth": True,
                            "include_subpath": True,
                        },
                        {"path": "/laya/v1/systemone", "target": gateway.upstream_url, "auth": True},
                    ],
                },
                "litellm_settings": {
                    **object_value(config["litellm_settings"]),
                    "overwrite_user_with_key_hash": True,
                },
            }
        )
    )
    overrides: Final = {"PROXY_BASE_URL": RESOURCE, "LITELLM_OAUTH_ADMIN_REDIRECT_URIS": CALLBACK}
    with (
        owned_proxy(gateway, tmp_path, overrides, config=path) as candidate,
        candidate.scenario() as scenario,
        httpx.Client(base_url=candidate.upstream_url, timeout=5, trust_env=False) as upstream,
    ):
        allowed: Final = scenario.model(model="openai/gpt-6.1-sol", input_cost_per_token=0.001, num_retries=0)
        blocked: Final = scenario.model(model="openai/gpt-6.1-sol", input_cost_per_token=0.001, num_retries=0)
        admin: Final = scenario.user(user_role="proxy_admin", models=[allowed], max_budget=1)
        issued: Final = candidate.client.post("/token", data=_authorize(candidate, admin))
        assert issued.status_code == 200, issued.text
        token: Final = string_value(JSON_OBJECT.validate_json(issued.content)["access_token"])
        assert candidate.request("GET", "/v1/chat/completions/health").status_code == 200
        for route in ("/v1/chat/completions/health", "/v1/chat/completions", "/laya/v1/systemone"):
            assert candidate.request("GET", route, key=token).status_code == 403
        assert candidate.request("POST", "/laya/v1/systemone", {}, key=token).status_code == 403
        body: Final = {
            "model": allowed,
            "messages": [{"role": "user", "content": "policy check"}],
            "user": admin,
            "cache": {"no-cache": True},
        }
        for model, fallbacks, route, status in (
            (allowed, [], "/v1/chat/completions", 200),
            (blocked, [], "/v1/chat/completions", 403),
            (allowed, [blocked], "/v1/chat/completions", 403),
            (allowed, [allowed], "/v1/chat/completions", 200),
            (allowed, [], f"/openai/deployments/{allowed}/chat/completions", 200),
            (blocked, [], f"/openai/deployments/{blocked}/chat/completions", 403),
        ):
            upstream.get("/__observations").raise_for_status()
            response: Final = candidate.request(
                "POST", route, {**body, "model": model, "fallbacks": fallbacks}, key=token
            )
            assert response.status_code == status, response.text
            requests: Final = object_value(upstream.get("/__observations").json())["requests"]
            assert isinstance(requests, list) and len(requests) == int(status == 200), requests
            if requests:
                assert object_value(object_value(requests[0])["body"])["user"] == admin
        eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_UserTable" WHERE user_id=%s', (admin,)),
            lambda rows: bool(rows) and float(str(rows[0]["spend"])) > 0,
            seconds=70,
        )
        for budget in (
            {"model_max_budget": {allowed: {"max_budget": 0, "budget_duration": "1d"}}},
            {"model_max_budget": {}, "max_budget": 0},
        ):
            candidate.post("/user/update", {"user_id": admin, **budget})
            denied: Final = candidate.request("POST", "/v1/chat/completions", body, key=token)
            assert denied.status_code == 422 and denied.json()["error"]["type"] == "budget_exceeded", denied.text
            assert upstream.get("/__observations").json()["requests"] == []
