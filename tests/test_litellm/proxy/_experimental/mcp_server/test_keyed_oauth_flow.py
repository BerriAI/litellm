import time
from typing import Final

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from litellm.proxy._experimental.mcp_server.gateway_dcr_flow import _seal
from litellm.proxy._experimental.mcp_server.keyed_oauth_flow import (
    KEYED_FLOW_PREFIX,
    open_keyed_oauth_flow,
    start_keyed_oauth_flow,
)
from litellm.proxy._types import UserAPIKeyAuth, hash_token


def _request(host: str = "localhost:4000", key: str | None = "sk-owned") -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "path": "/mcp",
            "headers": [
                (b"host", host.encode()),
                *(([(b"x-litellm-api-key", f"Bearer {key}".encode())]) if key else []),
            ],
        }
    )


@pytest.fixture(autouse=True)
def signing_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "keyed-flow-unit-test-salt")
    monkeypatch.delenv("PROXY_BASE_URL", raising=False)
    monkeypatch.delenv("SERVER_ROOT_PATH", raising=False)


def test_connection_identity_survives_headerless_discovery() -> None:
    request: Final = _request()
    sealed: Final = start_keyed_oauth_flow(request, UserAPIKeyAuth(user_id="owner"), "server-one")
    opened: Final = open_keyed_oauth_flow(_request(key=None), sealed)
    assert opened.key_hash == hash_token("sk-owned")
    assert opened.user_id == "owner"
    assert opened.server_id == "server-one"
    assert opened.resource == "http://localhost:4000/mcp"
    assert "sk-owned" not in sealed and "owner" not in sealed
    second: Final = open_keyed_oauth_flow(
        request, start_keyed_oauth_flow(request, UserAPIKeyAuth(user_id="owner"), "server-one")
    )
    assert opened.jti != second.jti


@pytest.mark.parametrize("fault", ("tampered", "foreign_resource", "expired", "wrong_type"))
def test_untrusted_connection_identity_cannot_authorize(fault: str) -> None:
    request: Final = _request()
    sealed: Final = start_keyed_oauth_flow(request, UserAPIKeyAuth(user_id="owner"), "server-one")
    opened: Final = open_keyed_oauth_flow(request, sealed)
    value: Final = (
        sealed[:40] + ("A" if sealed[40] != "A" else "B") + sealed[41:]
        if fault == "tampered"
        else _seal(KEYED_FLOW_PREFIX, opened.model_copy(update={"exp": 0}))
        if fault == "expired"
        else _seal(KEYED_FLOW_PREFIX, UserAPIKeyAuth(user_id="owner"))
        if fault == "wrong_type"
        else sealed
    )
    with pytest.raises(HTTPException) as error:
        open_keyed_oauth_flow(_request("other.example") if fault == "foreign_resource" else request, value)
    assert error.value.status_code == 400


@pytest.mark.parametrize("missing", ("key", "owner"))
def test_keyed_authorization_requires_an_owned_key(missing: str) -> None:
    with pytest.raises(HTTPException) as error:
        start_keyed_oauth_flow(
            _request(key=None if missing == "key" else "sk-owned"),
            UserAPIKeyAuth(user_id=None if missing == "owner" else "owner"),
            "server-one",
        )
    assert error.value.status_code == 403
    assert "assigned to a user" in error.value.detail


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ("none", "client", "pkce", "resource", "replayed", "storage"))
async def test_headerless_token_exchange_retains_original_key(monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    import base64
    import hashlib
    import json
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import httpx

    import litellm
    from litellm.caching.llm_caching_handler import LLMClientCache
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, httpxSpecialProvider
    from litellm.proxy._experimental.mcp_server import (
        bridge_token_flow,
        discoverable_endpoints,
        keyed_oauth_flow,
        mcp_server_manager,
    )
    from litellm.types.mcp import MCPAuth, MCPTransport
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    monkeypatch.setenv("PROXY_BASE_URL", "http://localhost:4000")
    server: Final = MCPServer(
        server_id="keyed-test-server",
        name="keyed-test-server",
        alias="keyed-test-server",
        transport=MCPTransport.http,
        auth_type=MCPAuth.oauth2,
        oauth2_flow="authorization_code",
        client_id="configured-client",
        client_secret="configured-secret",
        authorization_url="https://upstream.example/authorize",
        token_url="https://upstream.example/token",
    )
    registry: Final = mcp_server_manager.global_mcp_server_manager.registry
    monkeypatch.setitem(registry, server.server_id, server)
    verifier: Final = "v" * 43
    request: Final = _request(key=None)
    flow: Final = keyed_oauth_flow.KeyedOAuthFlow(
        key_hash=hash_token("sk-owned"),
        user_id="owner",
        server_id=server.server_id,
        resource="http://localhost:4000/mcp",
        jti="flow-id",
        exp=int(time.time()) + 600,
    )
    binding: Final = keyed_oauth_flow.KeyedAuthorization(
        flow=flow,
        client_id="registered-client",
        redirect_uri="http://localhost:8787/callback",
        state="client-state",
        code_challenge=base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode(),
    )
    store: Final = SimpleNamespace(
        get=AsyncMock(
            return_value=SimpleNamespace(
                binding_b64=_seal(keyed_oauth_flow.KEYED_BINDING_PREFIX, binding),
                status="code",
                expires_at=datetime.now(timezone.utc) + timedelta(minutes=2),
            )
        ),
        transition=AsyncMock(return_value=fault != "replayed"),
        revoke=AsyncMock(),
    )
    monkeypatch.setattr(keyed_oauth_flow, "_store", lambda: store)
    original_key: Final = UserAPIKeyAuth(api_key=flow.key_hash, user_id="owner", models=["limited-model"])
    monkeypatch.setattr(
        bridge_token_flow,
        "_reload_active_key_by_hash",
        AsyncMock(return_value=bridge_token_flow._ResolvedKey(flow.key_hash, original_key)),
    )
    monkeypatch.setattr(bridge_token_flow, "can_store_oauth_credential", AsyncMock(return_value=True))
    permission: Final = AsyncMock(return_value=True)
    monkeypatch.setattr(discoverable_endpoints, "can_store_oauth_credential", permission)
    stored: Final = AsyncMock(side_effect=RuntimeError("persistence failed") if fault == "storage" else None)
    monkeypatch.setattr(discoverable_endpoints, "_store_per_user_token_server_side", stored)
    clients: Final = LLMClientCache()
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", clients)

    def upstream_response(outbound: httpx.Request) -> httpx.Response:
        assert outbound.url == server.token_url
        assert b"code=real-upstream-code" in outbound.content
        assert b"client_id=configured-client" in outbound.content
        assert "x-litellm-api-key" not in outbound.headers
        return httpx.Response(
            200, json={"access_token": "upstream-token", "refresh_token": "upstream-refresh", "token_type": "Bearer"}
        )

    code: Final = _seal(
        keyed_oauth_flow.KEYED_CODE_PREFIX,
        keyed_oauth_flow.KeyedCode(grant_id=flow.jti, upstream_code="real-upstream-code", exp=int(time.time()) + 120),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream_response)) as transport:
        upstream: Final = AsyncHTTPHandler()
        await upstream.client.aclose()
        upstream.client = transport
        clients.set_cache("async_httpx_client" + httpxSpecialProvider.Oauth2Check, upstream)
        response: Final = await discoverable_endpoints.token_endpoint(
            request=request,
            grant_type="authorization_code",
            code=code,
            redirect_uri=binding.redirect_uri,
            client_id="foreign-client" if fault == "client" else binding.client_id,
            code_verifier="wrong" if fault == "pkce" else verifier,
            resource="https://foreign.example/mcp" if fault == "resource" else flow.resource,
            client_secret=None,
            refresh_token=None,
            scope=None,
            mcp_server_name=None,
        )
    if fault == "none":
        assert response.status_code == 200
        result: Final = json.loads(response.body)
        assert result["access_token"].startswith(keyed_oauth_flow.KEYED_ACCESS_PREFIX)
        assert result["refresh_token"].startswith(keyed_oauth_flow.KEYED_REFRESH_PREFIX)
        assert "upstream-token" not in response.body.decode()
        stored.assert_awaited_once()
        assert stored.call_args.kwargs["user_id"] == "owner"
        assert stored.call_args.kwargs["require_persistence"] is True
        assert permission.call_args.args[1] is original_key
        assert permission.call_args.args[1].models == ["limited-model"]
    elif fault == "storage":
        assert response.status_code == 503
        assert "access_token" not in json.loads(response.body)
    else:
        assert response.status_code == 400
        stored.assert_not_awaited()
        if fault == "replayed":
            store.revoke.assert_awaited_once_with(flow.jti)


@pytest.fixture
def active_grant(monkeypatch: pytest.MonkeyPatch):
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from litellm.proxy._experimental.mcp_server import keyed_oauth_flow

    flow: Final = keyed_oauth_flow.KeyedOAuthFlow(
        key_hash=hash_token("sk-owned"),
        user_id="owner",
        server_id="server-one",
        resource="http://localhost:4000/mcp",
        jti="grant-one",
        exp=1,
    )
    binding: Final = keyed_oauth_flow.KeyedAuthorization(
        flow=flow,
        client_id="client-one",
        redirect_uri="http://localhost:8787/callback",
        state="state",
        code_challenge="c" * 43,
    )
    row: Final = SimpleNamespace(
        binding_b64=_seal(keyed_oauth_flow.KEYED_BINDING_PREFIX, binding),
        status="active",
        expires_at=datetime.now(timezone.utc) + timedelta(days=1),
    )
    store: Final = SimpleNamespace(
        get=AsyncMock(return_value=row), transition=AsyncMock(return_value=True), revoke=AsyncMock()
    )
    monkeypatch.setattr(keyed_oauth_flow, "_store", lambda: store)
    return binding, row, store


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ("none", "wrong_key", "wrong_owner", "no_key", "expired", "refresh_as_access", "prefix_swap", "revoked")
)
async def test_keyed_access_never_replaces_or_widens_key_admission(
    monkeypatch: pytest.MonkeyPatch, active_grant, fault: str
) -> None:
    from litellm.proxy._experimental.mcp_server import keyed_oauth_flow
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager

    binding, row, store = active_grant
    token: Final = _seal(
        keyed_oauth_flow.KEYED_REFRESH_PREFIX if fault == "refresh_as_access" else keyed_oauth_flow.KEYED_ACCESS_PREFIX,
        keyed_oauth_flow.KeyedToken(
            kind="refresh" if fault in ("refresh_as_access", "prefix_swap") else "access",
            grant_id=binding.flow.jti,
            jti="access-one",
            exp=1 if fault == "expired" else int(time.time()) + 600,
        ),
    )
    request: Final = _request(key=None if fault == "no_key" else "sk-foreign" if fault == "wrong_key" else "sk-owned")
    request.scope["headers"].append((b"authorization", f"Bearer {token}".encode()))
    original: Final = UserAPIKeyAuth(user_id="other" if fault == "wrong_owner" else "owner", models=["limited-model"])
    original.via_virtual_key = fault != "no_key"
    if fault == "revoked":
        row.status = "revoked"
    if fault == "none":
        admitted: Final = await keyed_oauth_flow.validate_keyed_bearer(request, original)
        assert admitted.models == ["limited-model"]
        assert admitted.via_virtual_key
        assert not admitted.mcp_admitted_user_subject
        assert MCPServerManager._admitted_session_resource_scope(admitted) == "server-one"
        assert original.mcp_session_resource_server_id is None
    else:
        with pytest.raises(HTTPException):
            await keyed_oauth_flow.validate_keyed_bearer(request, original)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ("none", "replayed", "missing_upstream", "other_client", "access_as_refresh"))
async def test_keyed_refresh_uses_durable_grant_after_initial_flow_expires(
    monkeypatch: pytest.MonkeyPatch, active_grant, fault: str
) -> None:
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from litellm.proxy._experimental.mcp_server import bridge_token_flow, keyed_oauth_flow, mcp_server_manager

    binding, row, store = active_grant
    token: Final = _seal(
        keyed_oauth_flow.KEYED_REFRESH_PREFIX,
        keyed_oauth_flow.KeyedToken(
            kind="access" if fault == "access_as_refresh" else "refresh",
            grant_id=binding.flow.jti,
            jti="refresh-one",
            exp=int(time.time()) + 600,
        ),
    )
    auth: Final = UserAPIKeyAuth(user_id="owner")
    monkeypatch.setattr(
        bridge_token_flow,
        "_reload_active_key_by_hash",
        AsyncMock(return_value=bridge_token_flow._ResolvedKey(binding.flow.key_hash, auth)),
    )
    monkeypatch.setattr(bridge_token_flow, "_key_owner_scim_deactivated", AsyncMock(return_value=False))
    monkeypatch.setattr(bridge_token_flow, "can_store_oauth_credential", AsyncMock(return_value=True))
    server: Final = SimpleNamespace(server_id="server-one", is_gateway_managed_oauth2=True, needs_user_oauth_token=True)
    manager: Final = SimpleNamespace(
        get_mcp_server_by_id=lambda server_id: server,
        has_user_oauth_token=AsyncMock(return_value=fault != "missing_upstream"),
    )
    monkeypatch.setattr(mcp_server_manager, "global_mcp_server_manager", manager)
    store.transition.return_value = fault != "replayed"
    response: Final = await keyed_oauth_flow.exchange_keyed_token(
        _request(key=None),
        "refresh_token",
        None,
        None,
        "other" if fault == "other_client" else binding.client_id,
        None,
        token,
        binding.flow.resource,
        None,
    )
    if fault == "none":
        assert response.status_code == 200
        tokens: Final = json.loads(response.body)
        assert tokens["refresh_token"] != token
        assert store.transition.call_args.kwargs["expected_hash"] == hash_token(token)
        assert store.transition.call_args.kwargs["refresh_hash"] == hash_token(tokens["refresh_token"])
    else:
        assert response.status_code == 400
        if fault in ("replayed", "missing_upstream"):
            store.revoke.assert_awaited_once_with(binding.flow.jti)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ("none", "csrf", "cookie", "origin", "key", "attempts", "replayed", "server", "missing")
)
async def test_browser_confirmation_requires_same_key_and_browser(
    monkeypatch: pytest.MonkeyPatch, active_grant, fault: str
) -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from starlette.datastructures import FormData
    from fastapi.responses import RedirectResponse
    from litellm.proxy._experimental.mcp_server import keyed_oauth_flow, discoverable_endpoints, mcp_server_manager

    binding, row, store = active_grant
    row.status = "pending"
    store.attempt = AsyncMock(return_value=fault != "attempts")
    store.transition.return_value = fault != "replayed"
    auth: Final = UserAPIKeyAuth(user_id=binding.flow.user_id)
    verify_key: Final = AsyncMock(return_value=auth)
    monkeypatch.setattr(keyed_oauth_flow, "_active_key", verify_key)
    upstream: Final = AsyncMock(return_value=RedirectResponse("https://upstream.example/authorize"))
    monkeypatch.setattr(discoverable_endpoints, "authorize_with_server", upstream)
    server: Final = SimpleNamespace(
        is_gateway_managed_oauth2=True, needs_user_oauth_token=True, client_id="upstream-client"
    )
    monkeypatch.setattr(
        mcp_server_manager.global_mcp_server_manager,
        "get_mcp_server_by_id",
        lambda _: None if fault == "server" else server,
    )
    browser: Final = keyed_oauth_flow.KeyedBrowser(
        grant_id=binding.flow.jti, csrf="browser-secret", exp=int(time.time()) + 600
    )
    cookie: Final = _seal(keyed_oauth_flow.KEYED_BROWSER_PREFIX, browser)
    request: Final = _request(key=None)
    request.scope["headers"].extend(
        [
            (
                b"cookie",
                f"{keyed_oauth_flow._browser_cookie(binding.flow.jti)}={'broken' if fault == 'cookie' else cookie}".encode(),
            ),
            (b"origin", b"https://foreign.example" if fault == "origin" else b"http://localhost:4000"),
        ]
    )
    request._form = FormData(
        {
            "grant_id": binding.flow.jti,
            "csrf": "wrong" if fault == "csrf" else browser.csrf,
            **({} if fault == "missing" else {"api_key": "wrong" if fault == "key" else "Bearer sk-owned"}),
        }
    )
    if fault == "none":
        response: Final = await keyed_oauth_flow.confirm_keyed_authorization(request)
        assert response.headers["location"] == "https://upstream.example/authorize"
        assert "Max-Age=0" in response.headers["set-cookie"]
        assert upstream.call_args.kwargs["keyed_auth"] is auth
        assert upstream.call_args.kwargs["keyed_grant_id"] == binding.flow.jti
        assert upstream.call_args.kwargs["redirect_uri"] == binding.redirect_uri
        assert upstream.call_args.kwargs["code_challenge"] == binding.code_challenge
    else:
        with pytest.raises(HTTPException) as rejected:
            await keyed_oauth_flow.confirm_keyed_authorization(request)
        assert rejected.value.status_code in (400, 403)
        upstream.assert_not_awaited()
        if fault in ("csrf", "cookie", "origin", "missing"):
            store.attempt.assert_not_awaited()
        if fault == "key":
            verify_key.assert_not_awaited()
            store.transition.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ("none", "deleted_key", "storage_unavailable", "changed_owner", "scim", "denied"))
async def test_original_key_policy_is_rechecked_before_credential_write(
    monkeypatch: pytest.MonkeyPatch, active_grant, fault: str
) -> None:
    from unittest.mock import AsyncMock
    from litellm.proxy._experimental.mcp_server import bridge_token_flow, keyed_oauth_flow

    binding, _, _ = active_grant
    key: Final = UserAPIKeyAuth(
        user_id="other" if fault == "changed_owner" else binding.flow.user_id, models=["restricted"]
    )
    resolved: Final = (
        "no_active_key"
        if fault == "deleted_key"
        else "db_unavailable"
        if fault == "storage_unavailable"
        else bridge_token_flow._ResolvedKey(binding.flow.key_hash, key)
    )
    reload_key: Final = AsyncMock(return_value=resolved)
    policy: Final = AsyncMock(return_value=fault != "denied")
    monkeypatch.setattr(bridge_token_flow, "_reload_active_key_by_hash", reload_key)
    monkeypatch.setattr(bridge_token_flow, "_key_owner_scim_deactivated", AsyncMock(return_value=fault == "scim"))
    monkeypatch.setattr(bridge_token_flow, "can_store_oauth_credential", policy)
    if fault == "none":
        assert await keyed_oauth_flow._active_key(_request(key=None), binding) is key
        assert policy.call_args.args[1] is key
        assert policy.call_args.args[2] == binding.flow.server_id
    else:
        with pytest.raises(HTTPException) as rejected:
            await keyed_oauth_flow._active_key(_request(key=None), binding)
        assert rejected.value.status_code == (503 if fault == "storage_unavailable" else 403)
    reload_key.assert_awaited_once_with(binding.flow.key_hash)


@pytest.mark.asyncio
@pytest.mark.parametrize("replayed", (False, True))
async def test_callback_is_bound_to_single_durable_grant(
    monkeypatch: pytest.MonkeyPatch, active_grant, replayed: bool
) -> None:
    from unittest.mock import AsyncMock
    from litellm.proxy._experimental.mcp_server import keyed_oauth_flow

    binding, _, store = active_grant
    monkeypatch.setattr(keyed_oauth_flow, "_active_key", AsyncMock(return_value=UserAPIKeyAuth(user_id="owner")))
    store.transition.return_value = not replayed
    if replayed:
        with pytest.raises(HTTPException) as rejected:
            await keyed_oauth_flow.complete_keyed_callback(_request(), binding.flow.jti, "upstream-code")
        assert rejected.value.status_code == 400
    else:
        code: Final = await keyed_oauth_flow.complete_keyed_callback(_request(), binding.flow.jti, "upstream-code")
        opened: Final = keyed_oauth_flow._open_sealed(
            code, keyed_oauth_flow.KEYED_CODE_PREFIX, keyed_oauth_flow.KeyedCode, "test"
        )
        assert opened.grant_id == binding.flow.jti
        assert opened.upstream_code == "upstream-code"
        assert store.transition.call_args.args[1:3] == ("authorizing", "code")
        assert store.transition.call_args.kwargs["code_hash"] == hash_token(code)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ("none", "client", "resource", "missing", "malformed", "access"))
async def test_revocation_cannot_target_another_clients_grant(active_grant, fault: str) -> None:
    from litellm.proxy._experimental.mcp_server import keyed_oauth_flow

    binding, _, store = active_grant
    token: Final = _seal(
        keyed_oauth_flow.KEYED_REFRESH_PREFIX,
        keyed_oauth_flow.KeyedToken(
            kind="access" if fault == "access" else "refresh",
            grant_id=binding.flow.jti,
            jti="refresh",
            exp=int(time.time()) + 600,
        ),
    )
    if fault == "missing":
        store.get.return_value = None
    response: Final = await keyed_oauth_flow.revoke_keyed_token(
        _request(host="foreign.example" if fault == "resource" else "localhost:4000"),
        "broken" if fault == "malformed" else token,
        "other" if fault == "client" else binding.client_id,
    )
    assert response.status_code == (400 if fault in ("client", "resource") else 200)
    if fault == "none":
        store.revoke.assert_awaited_once_with(binding.flow.jti)
    else:
        store.revoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_token_storage_outage_returns_safe_oauth_error(active_grant) -> None:
    import json
    from litellm.proxy._experimental.mcp_server import keyed_oauth_flow

    binding, _, store = active_grant
    store.get.side_effect = RuntimeError("private-database-connection-string")
    refresh: Final = _seal(
        keyed_oauth_flow.KEYED_REFRESH_PREFIX,
        keyed_oauth_flow.KeyedToken(
            kind="refresh", grant_id=binding.flow.jti, jti="refresh", exp=int(time.time()) + 600
        ),
    )
    response: Final = await keyed_oauth_flow.exchange_keyed_token(
        _request(key=None), "refresh_token", None, None, binding.client_id, None, refresh, binding.flow.resource, None
    )
    assert response.status_code == 503
    assert json.loads(response.body)["error"] == "server_error"
    assert b"private-database" not in response.body
