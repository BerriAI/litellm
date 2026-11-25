from collections.abc import Iterator, Mapping
from datetime import datetime, timedelta, timezone
from http.cookies import SimpleCookie
from typing import Final
from unittest.mock import AsyncMock

import httpx
import jwt
import pytest
import time_machine
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
from litellm.integrations.lens.proxy.buffering import BufferBudget
from litellm.integrations.lens.proxy.model_gateway import GatewaySetupRequest
from litellm.integrations.lens.proxy.pages import (
    allow_page,
    connect_model_gateway,
    dashboard_session,
    lens_ui_asset,
    page_auth,
    router,
    same_origin,
    ui_mount,
)
from litellm.integrations.lens.remote import LensConnection
from litellm.proxy import proxy_server
from litellm.proxy._types import LiteLLM_UserTable, LitellmUserRoles, UserAPIKeyAuth, hash_token
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.redis.stores.auth_cache import AuthObject
from tests.unit.proxy._prisma import prisma_with_empty_config
from tests.unit.proxy.rust_auth_stores import FakeKeyStore, install_rust_auth_stores

SECRET: Final = "gateway-delegated-identity-fixture-32"
UNSUPPORTED: Final = "The connected Lens does not serve a UI this gateway can show. Update Lens to use it here"

LENS: Final = LensConnection("http://lens.test", SECRET)
SESSION_NOW: Final = datetime(2026, 11, 25, 12, tzinfo=timezone.utc)
SESSION_EXPIRY: Final = SESSION_NOW + timedelta(hours=1)
SESSION_KEY: Final = "sk-dashboard-session-fixture"


def _dashboard_token(**claims: object) -> str:
    payload: Final = {
        "user_id": "alice",
        "key": SESSION_KEY,
        "login_method": "sso",
        "exp": int(SESSION_EXPIRY.timestamp()),
        **claims,
    }
    return jwt.encode(
        {name: value for name, value in payload.items() if value is not None},
        SECRET,
        algorithm="HS256",
    )


@pytest.fixture
def session_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    stores: Final = install_rust_auth_stores(monkeypatch)
    monkeypatch.setattr(
        stores,
        "key",
        FakeKeyStore(
            rows={
                hash_token(key): {
                    "token": hash_token(key),
                    "user_id": user_id,
                    "team_id": UI_SESSION_TOKEN_TEAM_ID,
                    "expires": expiry,
                }
                for key, user_id, expiry in (
                    (SESSION_KEY, "alice", SESSION_EXPIRY),
                    ("sk-other-account", "bob", SESSION_EXPIRY),
                    ("sk-expired-session", "alice", SESSION_NOW - timedelta(seconds=1)),
                )
            }
        ),
    )
    cache: Final = UserApiKeyCache()
    for user_id in ("alice", "bob"):
        cache.set_cache(
            AuthObject.user(user_id),
            LiteLLM_UserTable(user_id=user_id, user_role="internal_user"),
            model_type=LiteLLM_UserTable,
        )
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    monkeypatch.setattr(proxy_server, "prisma_client", prisma_with_empty_config())
    monkeypatch.setattr(proxy_server, "master_key", SECRET)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "user_custom_auth", None)
    monkeypatch.setattr(proxy_server, "llm_router", None)
    monkeypatch.delenv("PROXY_BASE_URL", raising=False)
    app: Final = FastAPI()
    app.include_router(router)
    with time_machine.travel(SESSION_NOW, tick=False), TestClient(app, base_url="https://gateway.test") as client:
        yield client


@pytest.mark.asyncio
async def test_dashboard_token_restores_the_same_login_for_lens_without_extending_expiry(
    session_client: TestClient,
) -> None:
    token: Final = _dashboard_token()
    response: Final = session_client.post(
        "/lens/session", json={"token": token}, headers={"sec-fetch-site": "same-origin"}
    )
    assert response.status_code == 204, response.text
    assert response.content == b""
    assert response.headers["cache-control"] == "private, no-store"
    cookie: Final = SimpleCookie(response.headers["set-cookie"])["token"]
    assert (cookie.value, cookie["path"], cookie["samesite"], cookie["secure"]) == (token, "/", "lax", True)
    request: Final = _request("GET", {"cookie": f"token={cookie.value}"}, scheme="https")
    auth: Final = await page_auth(request)
    assert auth.user_id == "alice"
    with time_machine.travel(SESSION_EXPIRY, tick=False), pytest.raises(HTTPException) as error:
        await page_auth(request)
    assert error.value.status_code == 401


@pytest.mark.parametrize(
    "claims",
    (
        {"exp": int(SESSION_NOW.timestamp())},
        {"exp": None},
        {"type": "byok_session"},
        {"login_method": "api_key"},
        {"key": "sk-revoked-session"},
        {"key": "sk-expired-session"},
        {"user_id": "bob"},
    ),
    ids=("expired-jwt", "missing-expiry", "byok-token", "non-login", "revoked-key", "expired-key", "wrong-user"),
)
def test_invalid_dashboard_sessions_cannot_restore_a_cookie(
    session_client: TestClient, claims: dict[str, object]
) -> None:
    response: Final = session_client.post(
        "/lens/session", json={"token": _dashboard_token(**claims)}, headers={"sec-fetch-site": "same-origin"}
    )
    assert response.status_code == 401, response.text
    assert "set-cookie" not in response.headers


def test_tampered_dashboard_token_cannot_restore_a_cookie(session_client: TestClient) -> None:
    response: Final = session_client.post(
        "/lens/session", json={"token": _dashboard_token() + "invalid"}, headers={"sec-fetch-site": "same-origin"}
    )
    assert response.status_code == 401
    assert "set-cookie" not in response.headers


def test_session_rejection_does_not_log_the_dashboard_token(
    session_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure: Final = AsyncMock(return_value=None)
    monkeypatch.setattr(proxy_server.proxy_logging_obj, "post_call_failure_hook", failure)
    token: Final = _dashboard_token(key="sk-revoked-session")
    response: Final = session_client.post(
        "/lens/session", json={"token": token}, headers={"sec-fetch-site": "same-origin"}
    )
    assert response.status_code == 401
    failure.assert_awaited()
    assert token not in repr(failure.call_args)


def test_custom_auth_can_read_the_sanitized_session_request(
    session_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def custom_auth(request: Request, api_key: str) -> UserAPIKeyAuth:
        assert await request.json() == {}
        assert api_key == SESSION_KEY
        return UserAPIKeyAuth(user_id="alice", user_role=LitellmUserRoles.INTERNAL_USER)

    monkeypatch.setattr(proxy_server, "user_custom_auth", custom_auth)
    response: Final = session_client.post(
        "/lens/session", json={"token": _dashboard_token()}, headers={"sec-fetch-site": "same-origin"}
    )
    assert response.status_code == 204, response.text
    assert session_client.cookies.get("token") == _dashboard_token()


@pytest.mark.parametrize("site", ("cross-site", "same-site", "none", ""))
def test_other_origins_cannot_restore_a_dashboard_session(session_client: TestClient, site: str) -> None:
    response: Final = session_client.post(
        "/lens/session", json={"token": _dashboard_token()}, headers={"sec-fetch-site": site}
    )
    assert response.status_code == 403
    assert "set-cookie" not in response.headers


def test_a_stale_dashboard_tab_cannot_replace_another_live_account(session_client: TestClient) -> None:
    current: Final = _dashboard_token(user_id="bob", key="sk-other-account")
    session_client.cookies.set("token", current)
    response: Final = session_client.post(
        "/lens/session", json={"token": _dashboard_token()}, headers={"sec-fetch-site": "same-origin"}
    )
    assert response.status_code == 409
    assert "set-cookie" not in response.headers
    assert session_client.cookies.get("token") == current


@pytest.mark.parametrize(
    ("current_claims", "status"),
    (({}, 204), ({"exp": int(SESSION_NOW.timestamp())}, 401), ({"key": "sk-revoked-session"}, 401)),
)
@pytest.mark.parametrize(
    "stored_claims",
    ({"exp": int(SESSION_NOW.timestamp())}, {"key": "sk-revoked-session"}),
)
def test_stale_storage_uses_only_a_live_browser_cookie(
    session_client: TestClient,
    current_claims: dict[str, object],
    stored_claims: dict[str, object],
    status: int,
) -> None:
    current: Final = _dashboard_token(**current_claims)
    session_client.cookies.set("token", current)
    response: Final = session_client.post(
        "/lens/session", json={"token": _dashboard_token(**stored_claims)}, headers={"sec-fetch-site": "same-origin"}
    )
    assert response.status_code == status
    assert "set-cookie" not in response.headers
    assert session_client.cookies.get("token") == current


@pytest.mark.parametrize("header", ("authorization", "x-litellm-api-key", "x-dashboard-key"))
def test_another_bearer_cannot_substitute_for_the_dashboard_session_key(
    session_client: TestClient, monkeypatch: pytest.MonkeyPatch, header: str
) -> None:
    monkeypatch.setattr(proxy_server, "general_settings", {"litellm_key_header_name": "x-dashboard-key"})
    response: Final = session_client.post(
        "/lens/session",
        json={"token": _dashboard_token(key="sk-revoked-session")},
        headers={"sec-fetch-site": "same-origin", header: f"Bearer {SESSION_KEY}"},
    )
    assert response.status_code == 400
    assert "set-cookie" not in response.headers


@pytest.mark.asyncio
async def test_ui_files_come_from_the_connected_lens_and_only_the_dashboard_may_frame_them() -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://lens.test/ui/_next/static/app%20v2.js"
        assert request.headers["x-lens-ui-mount"] == "/team-a/lens/ui"
        assert "authorization" not in request.headers
        return httpx.Response(
            200,
            content=b"console.log('lens')",
            headers={
                "content-type": "text/javascript",
                "cache-control": "no-cache",
                "x-content-type-options": "nosniff",
                "set-cookie": "lens_session=leaked",
                "x-internal": "private",
                "content-security-policy": "script-src 'self'; frame-ancestors 'none'; connect-src 'self'",
                "x-lens-ui-mount": "/team-a/lens/ui",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        response: Final = await lens_ui_asset("_next/static/app v2.js", LENS, client, ui_mount("/team-a/"))
    assert response.status_code == 200
    assert response.body == b"console.log('lens')"
    assert {name: value for name, value in response.headers.items() if name != "content-length"} == {
        "content-type": "text/javascript",
        "cache-control": "no-cache",
        "x-content-type-options": "nosniff",
        "x-frame-options": "SAMEORIGIN",
        "content-security-policy": "script-src 'self'; connect-src 'self'; frame-ancestors 'self'",
    }


@pytest.mark.parametrize(("root_path", "mount"), (("", "/lens/ui"), ("/", "/lens/ui"), ("/team-a", "/team-a/lens/ui")))
def test_ui_mount_follows_the_gateway_root_path(root_path: str, mount: str) -> None:
    assert ui_mount(root_path) == mount


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "reported", "detail"),
    (
        (404, 404, UNSUPPORTED),
        (401, 503, "Lens service is unavailable"),
        (302, 503, "Lens service is unavailable"),
        (500, 503, "Lens service is unavailable"),
    ),
    ids=("older_lens", "refused", "redirect", "broken"),
)
async def test_only_a_lens_with_no_ui_route_reads_as_older_and_its_body_is_dropped(
    status: int, reported: int, detail: str
) -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=b"lens internal detail", headers={"location": "http://lens.test/ui/"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        with pytest.raises(HTTPException) as error:
            await lens_ui_asset("", LENS, client, "/lens/ui")
    assert (error.value.status_code, error.value.detail) == (reported, detail)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ("../internal/spend", "_next/../../auth/session", "a/./b", "..\\internal"))
async def test_ui_path_cannot_reach_another_lens_route(path: str) -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Lens was asked for {request.url}")

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        with pytest.raises(HTTPException) as error:
            await lens_ui_asset(path, LENS, client, "/lens/ui")
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_ui_is_not_served_by_a_gateway_with_no_lens_connection() -> None:
    with pytest.raises(HTTPException) as error:
        await lens_ui_asset("", None, None, "/lens/ui")
    assert error.value.status_code == 404


@pytest.mark.asyncio
async def test_ui_reports_an_unreachable_lens_apart_from_an_older_one() -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Fixture Lens is down", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        with pytest.raises(HTTPException) as error:
            await lens_ui_asset("", LENS, client, "/lens/ui")
    assert error.value.status_code == 503


@pytest.mark.asyncio
async def test_ui_asset_is_charged_to_the_buffer_budget_until_it_is_sent() -> None:
    asset: Final = b"x" * 600

    def transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=asset, headers={"x-lens-ui-mount": "/lens/ui"})

    async def receive() -> dict[str, str]:
        return {"type": "http.disconnect"}

    async def send(message: Mapping[str, object]) -> None:
        return None

    budget: Final = BufferBudget(1000)
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        held: Final = await lens_ui_asset("", LENS, client, "/lens/ui", budget=budget)
        with pytest.raises(HTTPException) as busy:
            await lens_ui_asset("", LENS, client, "/lens/ui", budget=budget)
        assert (busy.value.status_code, busy.value.headers) == (503, {"Retry-After": "1"})
        await held({"type": "http", "method": "GET"}, receive, send)
        after_send: Final = await lens_ui_asset("", LENS, client, "/lens/ui", budget=budget)
    assert after_send.body == asset


@pytest.mark.asyncio
@pytest.mark.parametrize("named", (None, "/ui", "/other/lens/ui"), ids=("ignored", "its_own_path", "another_mount"))
async def test_ui_from_a_lens_that_does_not_mount_it_here_reads_as_older(named: str | None) -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'<script src="/ui/_next/static/app.js"></script>',
            headers={"x-lens-ui-mount": named} if named else {},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        with pytest.raises(HTTPException) as error:
            await lens_ui_asset("", LENS, client, "/lens/ui")
    assert (error.value.status_code, error.value.detail) == (404, UNSUPPORTED)


def _request(method: str, headers: Mapping[str, str], scheme: str = "http") -> Request:
    return Request(
        {
            "type": "http",
            "scheme": scheme,
            "server": ("gateway.test", 443 if scheme == "https" else 80),
            "method": method,
            "path": "/lens/datasets",
            "query_string": b"",
            "headers": [(name.encode(), value.encode()) for name, value in headers.items()],
        }
    )


@pytest.mark.parametrize(
    ("headers", "allowed"),
    (
        ({"sec-fetch-site": "same-origin"}, True),
        ({"sec-fetch-site": "same-site", "origin": "http://gateway.test", "host": "gateway.test"}, False),
        ({"sec-fetch-site": "cross-site"}, False),
        ({"origin": "http://gateway.test", "host": "gateway.test"}, True),
        ({"origin": "http://evil.test", "host": "gateway.test"}, False),
        ({"host": "gateway.test"}, False),
    ),
    ids=("same_origin", "sibling_site", "cross_site", "matching_origin", "other_origin", "no_browser_proof"),
)
def test_only_a_page_on_the_gateway_address_counts_as_same_origin(headers: Mapping[str, str], allowed: bool) -> None:
    assert same_origin(_request("POST", headers)) is allowed


@pytest.mark.parametrize(
    ("origin", "allowed"),
    (("https://gateway.test", True), ("http://gateway.test", False)),
    ids=("same_scheme", "plain_http_page"),
)
def test_a_page_on_another_scheme_of_the_gateway_host_is_not_same_origin(origin: str, allowed: bool) -> None:
    request: Final = _request("POST", {"origin": origin, "host": "gateway.test"}, scheme="https")
    assert same_origin(request) is allowed


async def _signed_in(request: Request) -> UserAPIKeyAuth | None:
    return UserAPIKeyAuth(user_id="alice")


async def _signed_out(request: Request) -> UserAPIKeyAuth | None:
    return None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "configured_url",
    (
        None,
        "not-a-url",
        "https://[broken",
        "https://user@gateway.test",
        "http://bad host",
        "https://gateway.test:99999",
    ),
)
async def test_connect_requires_a_configured_gateway_url(
    monkeypatch: pytest.MonkeyPatch, configured_url: str | None
) -> None:
    if configured_url is None:
        monkeypatch.delenv("PROXY_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("PROXY_BASE_URL", configured_url)
    monkeypatch.setenv("LITELLM_LENS_URL", "http://lens.test")
    monkeypatch.setenv("LITELLM_LENS_SERVICE_TOKEN", SECRET)
    monkeypatch.setenv("LENS_GATEWAY_SECRET", SECRET)
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)

    async def admin_key(**_headers: object) -> UserAPIKeyAuth:
        return UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)

    async def log_teams(_auth: UserAPIKeyAuth) -> tuple[str, ...]:
        return ()

    monkeypatch.setattr("litellm.integrations.lens.proxy.pages.user_api_key_auth", admin_key)
    request: Final = _request("POST", {"host": "request-host.test", "authorization": "Bearer fixture"})
    with pytest.raises(HTTPException) as error:
        await connect_model_gateway(request, GatewaySetupRequest(), log_teams)

    assert error.value.status_code == 503
    assert "PROXY_BASE_URL" in error.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "headers"),
    (
        ("GET", {}),
        ("POST", {"sec-fetch-site": "same-origin"}),
        ("DELETE", {"origin": "http://gateway.test", "host": "gateway.test"}),
    ),
    ids=("read", "dashboard_write", "dashboard_delete"),
)
async def test_the_dashboard_login_is_enough_for_pages_the_dashboard_loads(
    method: str, headers: Mapping[str, str]
) -> None:
    auth: Final = await page_auth(_request(method, headers), session=_signed_in)
    assert auth.user_id == "alice"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method", "headers"),
    (("POST", {}), ("PUT", {"sec-fetch-site": "cross-site"}), ("DELETE", {"origin": "http://evil.test"})),
    ids=("no_browser_proof", "cross_site", "other_origin"),
)
async def test_another_site_cannot_change_lens_with_the_dashboard_login(
    method: str, headers: Mapping[str, str]
) -> None:
    with pytest.raises(HTTPException) as error:
        await page_auth(_request(method, headers), session=_signed_in)
    assert error.value.status_code == 403


@pytest.mark.asyncio
async def test_pages_need_a_dashboard_login_when_no_key_is_sent() -> None:
    with pytest.raises(HTTPException) as error:
        await page_auth(_request("GET", {"sec-fetch-site": "same-origin"}), session=_signed_out)
    assert error.value.status_code == 401


@pytest.mark.asyncio
async def test_a_key_is_used_instead_of_the_dashboard_login_when_one_is_sent() -> None:
    async def keyed(request: Request, key: str) -> UserAPIKeyAuth:
        return UserAPIKeyAuth(user_id=f"owner-of-{key}")

    auth: Final = await page_auth(
        _request("POST", {"authorization": "Bearer sk-cli", "sec-fetch-site": "cross-site"}),
        session=_signed_in,
        key=keyed,
    )
    assert auth.user_id == "owner-of-Bearer sk-cli"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    (
        {"authorization": "Bearer sk-gateway-master"},
        {"x-litellm-api-key": "sk-gateway-master"},
        {"x-gateway-key": "Bearer sk-gateway-master"},
        {"api-key": "sk-gateway-master"},
        {"x-api-key": "sk-gateway-master"},
        {"x-goog-api-key": "sk-gateway-master"},
        {"ocp-apim-subscription-key": "sk-gateway-master"},
    ),
    ids=("authorization", "litellm", "configured_name", "azure", "anthropic", "google", "azure_apim"),
)
async def test_a_key_in_any_header_the_gateway_reads_keys_from_signs_the_caller_in(
    monkeypatch: pytest.MonkeyPatch, headers: Mapping[str, str]
) -> None:
    monkeypatch.setattr(proxy_server, "master_key", "sk-gateway-master")
    monkeypatch.setattr(proxy_server, "general_settings", {"litellm_key_header_name": "X-Gateway-Key"})

    auth: Final = await page_auth(_request("GET", headers), session=_signed_out)

    assert auth.user_role == LitellmUserRoles.PROXY_ADMIN


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "empty_header",
    (
        "authorization",
        "x-litellm-api-key",
        "x-gateway-key",
        "api-key",
        "x-api-key",
        "x-goog-api-key",
        "ocp-apim-subscription-key",
    ),
)
async def test_an_empty_key_header_does_not_replace_the_dashboard_login(
    monkeypatch: pytest.MonkeyPatch, empty_header: str
) -> None:
    monkeypatch.setattr(proxy_server, "general_settings", {"litellm_key_header_name": "X-Gateway-Key"})

    auth: Final = await page_auth(
        _request("GET", {empty_header: "", "sec-fetch-site": "same-origin"}), session=_signed_in
    )

    assert auth.user_id == "alice"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "empty_header", ("api-key", "x-goog-api-key", "ocp-apim-subscription-key", "x-litellm-api-key")
)
async def test_an_empty_key_header_does_not_hide_a_key_sent_in_another_header(
    monkeypatch: pytest.MonkeyPatch, empty_header: str
) -> None:
    monkeypatch.setattr(proxy_server, "master_key", "sk-gateway-master")
    monkeypatch.setattr(proxy_server, "general_settings", {})

    auth: Final = await page_auth(
        _request("GET", {empty_header: "", "x-api-key": "sk-gateway-master"}), session=_signed_out
    )

    assert auth.user_role == LitellmUserRoles.PROXY_ADMIN


def _cookie(user_id: str | None, session_key: str | None):
    def read(_: Request) -> tuple[str | None, str | None]:
        return user_id, session_key

    return read


@pytest.mark.asyncio
async def test_dashboard_session_runs_the_gateway_route_rules_on_the_cookie_login() -> None:
    seen: Final[list[str]] = []  # mutable-ok: records the key handed to the gateway's own auth

    async def key(_: Request, api_key: str) -> UserAPIKeyAuth:
        seen.append(api_key)
        return UserAPIKeyAuth(user_id="alice")

    auth: Final = await dashboard_session(_request("GET", {}), cookie=_cookie("alice", "sk-session"), key=key)

    assert auth is not None and auth.user_id == "alice"
    assert seen == ["Bearer sk-session"]


@pytest.mark.asyncio
async def test_dashboard_session_is_refused_when_the_gateway_route_rules_refuse_it() -> None:
    async def refused(_: Request, __: str) -> UserAPIKeyAuth:
        raise HTTPException(403, "this login may only change its password")

    assert await dashboard_session(_request("GET", {}), cookie=_cookie("alice", "sk-session"), key=refused) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_id", "session_key", "resolved"), [(None, None, "alice"), ("alice", "sk-session", "bob")]
)
async def test_dashboard_session_needs_a_cookie_login_that_resolves_to_its_own_user(
    user_id: str | None, session_key: str | None, resolved: str
) -> None:
    async def key(_: Request, __: str) -> UserAPIKeyAuth:
        return UserAPIKeyAuth(user_id=resolved)

    assert await dashboard_session(_request("GET", {}), cookie=_cookie(user_id, session_key), key=key) is None


@pytest.mark.parametrize(
    ("path", "role"),
    [
        ("admin/signals", LitellmUserRoles.INTERNAL_USER),
        ("admin/signals", LitellmUserRoles.INTERNAL_USER_VIEW_ONLY),
        ("admin/a-page-a-newer-lens-added", LitellmUserRoles.INTERNAL_USER),
        ("admin", LitellmUserRoles.INTERNAL_USER),
        ("admin/signals", None),
    ],
)
def test_lens_admin_pages_stop_at_the_gateway_for_everyone_but_admins(path: str, role: LitellmUserRoles | None) -> None:
    with pytest.raises(HTTPException) as refused:
        allow_page(path, UserAPIKeyAuth(user_id="alice", user_role=role))

    assert refused.value.status_code == 403


@pytest.mark.parametrize(
    ("path", "role"),
    [
        ("admin/signals", LitellmUserRoles.PROXY_ADMIN),
        ("admin/traces/signals", LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY),
        ("v1/traces", LitellmUserRoles.INTERNAL_USER),
        ("administrators", LitellmUserRoles.INTERNAL_USER),
        ("evals/admin", LitellmUserRoles.INTERNAL_USER),
        ("", LitellmUserRoles.INTERNAL_USER),
    ],
)
def test_admins_reach_lens_admin_pages_and_everyone_reaches_the_rest(path: str, role: LitellmUserRoles) -> None:
    assert allow_page(path, UserAPIKeyAuth(user_id="alice", user_role=role)) is None
