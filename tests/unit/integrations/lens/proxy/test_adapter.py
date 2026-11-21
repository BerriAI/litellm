from collections.abc import Mapping
from typing import Final

import httpx
import jwt
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from litellm.integrations.lens.proxy.adapter import (
    Connection,
    Identity,
    delegated_identity,
    forward,
    lens_request,
    login_redirect,
    login_user_name,
    password_change_redirect,
    request_body,
    service_connection,
    service_status,
)
from litellm.integrations.lens.remote import LensConnection
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

SECRET: Final = "gateway-delegated-identity-fixture-32"


@pytest.fixture
def connection() -> Connection:
    return Connection(LensConnection("http://lens.test", SECRET), SECRET)


@pytest.fixture
def identity() -> Identity:
    return Identity(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="user",
        team_id="team",
        org_id="org",
        token="token-hash",
        models=("allowed-model",),
        log_team_ids=("permitted-team",),
    )


@pytest.mark.parametrize(
    ("overrides", "available"),
    (
        ({}, True),
        ({"LENS_GATEWAY_SECRET": "short"}, False),
        ({"LENS_GATEWAY_SECRET": "x" * 513}, False),
        ({"LITELLM_LENS_URL": "file:///tmp/lens"}, False),
        ({"LITELLM_LENS_SERVICE_TOKEN": ""}, False),
    ),
    ids=("configured", "short_signing_secret", "long_signing_secret", "invalid_url", "missing_service_token"),
)
def test_connection_requires_valid_endpoint_and_both_secrets(overrides: Mapping[str, str], available: bool) -> None:
    configured: Final = Connection.from_env(
        {
            "LITELLM_LENS_URL": "http://lens.test",
            "LITELLM_LENS_SERVICE_TOKEN": SECRET,
            "LENS_GATEWAY_SECRET": SECRET,
            **overrides,
        }
    )
    assert (configured is not None) == available


def test_delegated_token_contains_only_authenticated_identity(connection: Connection, identity: Identity) -> None:
    token: Final = connection.identity_token(identity, 1000)
    assert token is not None
    decoded: Final = jwt.decode(
        token, SECRET, algorithms=["HS256"], audience="litellm-lens", issuer="litellm", options={"verify_exp": False}
    )
    assert decoded == {
        "iss": "litellm",
        "aud": "litellm-lens",
        "sub": "user",
        "iat": 1000,
        "exp": 1030,
        "identity": identity.model_dump(mode="json"),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "role",
    (
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
        LitellmUserRoles.INTERNAL_USER,
        LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
        None,
    ),
    ids=("admin", "admin_viewer", "user", "user_viewer", "default_user"),
)
async def test_delegated_identity_preserves_authenticated_lens_scope_and_log_permissions(
    role: LitellmUserRoles | None,
) -> None:
    async def log_team_lookup(auth: UserAPIKeyAuth) -> tuple[str, ...]:
        assert auth.user_id == "user"
        assert role not in (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)
        return ("permitted-team",)

    identity: Final = await delegated_identity(
        UserAPIKeyAuth(
            user_role=role,
            user_id="user",
            team_id="credential-team",
            org_id="org",
            token="token-hash",
            models=["model"],
        ),
        log_team_lookup,
    )
    assert identity == Identity(
        user_role=role or LitellmUserRoles.INTERNAL_USER,
        user_id="user",
        team_id="credential-team",
        org_id="org",
        token="token-hash",
        models=("model",),
        log_team_ids=()
        if role in (LitellmUserRoles.PROXY_ADMIN, LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY)
        else ("permitted-team",),
    )


@pytest.mark.asyncio
async def test_delegation_permission_failure_falls_back_to_own_user_scope() -> None:
    async def unavailable(auth: UserAPIKeyAuth) -> tuple[str, ...]:
        raise RuntimeError("Permission database unavailable")

    identity: Final = await delegated_identity(
        UserAPIKeyAuth(user_id="user", user_role=LitellmUserRoles.INTERNAL_USER), unavailable
    )
    assert identity.user_id == "user"
    assert identity.log_team_ids == ()


@pytest.mark.asyncio
async def test_forwarding_replaces_caller_headers_and_preserves_wire_response(
    connection: Connection,
    identity: Identity,
) -> None:
    async def receive() -> Mapping[str, object]:
        return {"type": "http.request", "body": b'{"case":"value"}', "more_body": False}

    def transport(request: httpx.Request) -> httpx.Response:
        assert request.url == "http://lens.test/lens/evals/runs?offset=2&tag=a&tag=b"
        assert request.content == b'{"case":"value"}'
        assert request.headers["x-lens-contract"] == "1"
        assert request.headers["idempotency-key"] == "dedupe"
        assert "cookie" not in request.headers
        assert "x-lens-internal" not in request.headers
        token: Final = request.headers["authorization"].removeprefix("Bearer ")
        assert jwt.decode(token, SECRET, algorithms=["HS256"], audience="litellm-lens", options={"verify_exp": False})[
            "identity"
        ] == identity.model_dump(mode="json")
        return httpx.Response(
            409,
            content=b'{"detail":"contract mismatch"}',
            headers={"content-type": "application/json", "retry-after": "2", "set-cookie": "unsafe"},
        )

    request: Final = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/lens/evals/runs",
            "query_string": b"offset=2&tag=a&tag=b",
            "headers": [
                (b"authorization", b"Bearer raw-key"),
                (b"x-lens-contract", b"1"),
                (b"idempotency-key", b"dedupe"),
                (b"cookie", b"private"),
                (b"x-lens-internal", b"forged"),
            ],
            "scheme": "http",
            "server": ("gateway.test", 80),
        },
        receive,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        response: Final = await forward(request, identity, "evals/runs", connection, client, 1000)
    assert response.status_code == 409
    assert response.body == b'{"detail":"contract mismatch"}'
    assert response.headers["retry-after"] == "2"
    assert "set-cookie" not in response.headers


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ("../internal/read", "foo/../bar", "foo\\bar"))
async def test_forwarding_cannot_escape_lens_namespace(connection: Connection, identity: Identity, path: str) -> None:
    request: Final = Request({"type": "http", "method": "GET", "headers": []})
    async with httpx.AsyncClient() as client:
        with pytest.raises(HTTPException) as error:
            await forward(request, identity, path, connection, client, 1000)
    assert error.value.status_code == 400


@pytest.mark.asyncio
@pytest.mark.parametrize("status", (401, 503))
async def test_unavailable_status_stays_disconnected(connection: Connection, status: int) -> None:
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(status))) as client:
        result: Final = await service_status(connection, client)
    assert result.protocol_version == 0
    assert not result.storage_ready


@pytest.mark.asyncio
async def test_status_preserves_public_contract_separately_from_worker_protocol(connection: Connection) -> None:
    payload: Final = {
        "storage_ready": True,
        "credentials_ready": True,
        "release": "test-release",
        "protocol_version": 7,
        "public_contract": 1,
    }
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))) as client:
        result: Final = await service_status(connection, client)
    assert result.model_dump() == payload


@pytest.mark.asyncio
async def test_trailing_slash_redirect_retains_gateway_prefix_and_query(
    connection: Connection, identity: Identity
) -> None:
    request: Final = Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/gateway/lens/datasets/",
            "query_string": b"offset=2",
            "headers": [],
            "scheme": "https",
            "server": ("gateway.test", 443),
        }
    )
    async with httpx.AsyncClient() as client:
        response: Final = await forward(request, identity, "datasets/", connection, client, 1000)
    assert response.status_code == 307
    assert response.headers["location"] == "https://gateway.test/gateway/lens/datasets?offset=2"


@pytest.mark.asyncio
async def test_duplicate_contract_headers_cannot_be_collapsed_to_a_supported_version(
    connection: Connection, identity: Identity
) -> None:
    request: Final = Request(
        {"type": "http", "method": "GET", "headers": [(b"x-lens-contract", b"2"), (b"x-lens-contract", b"1")]}
    )
    async with httpx.AsyncClient() as client:
        response: Final = await forward(request, identity, "datasets", connection, client, 1000)
    assert response.status_code == 409
    assert response.body == b'{"detail":"contract_version","code":"contract_version"}'


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", (4, 5), ids=("overflow", "exact_limit"))
async def test_streamed_body_bound_is_enforced_without_content_length(limit: int) -> None:
    chunks: Final = iter((b"ab", b"cde"))

    async def receive() -> Mapping[str, object]:
        chunk: Final = next(chunks)
        return {"type": "http.request", "body": chunk, "more_body": chunk == b"ab"}

    request: Final = Request({"type": "http", "method": "POST", "headers": []}, receive)
    if limit == 5:
        assert await request_body(request, limit) == b"abcde"
        return
    with pytest.raises(HTTPException) as error:
        await request_body(request, limit)
    assert error.value.status_code == 413


@pytest.mark.asyncio
@pytest.mark.parametrize("configured", (False, True), ids=("unconfigured", "missing_signing_secret"))
async def test_missing_connection_keeps_setup_readable_and_rejects_lens_requests(
    monkeypatch: pytest.MonkeyPatch, identity: Identity, configured: bool
) -> None:
    monkeypatch.delenv("LENS_GATEWAY_SECRET", raising=False)
    monkeypatch.setenv("LITELLM_LENS_SERVICE_TOKEN", SECRET)
    monkeypatch.setenv("LITELLM_LENS_URL", "http://lens.test" if configured else "")
    monkeypatch.setenv("LITELLM_LENS_PUBLIC_URL", "https://lens.example.test/")
    status: Final = await service_connection(identity)
    assert status.configured == configured
    assert not status.connected
    assert not status.status.storage_ready
    assert status.url == "https://lens.example.test"
    request: Final = Request({"type": "http", "method": "GET", "headers": []})
    with pytest.raises(HTTPException) as routed:
        await lens_request(request, identity, "datasets")
    assert routed.value.status_code == 503
    async with httpx.AsyncClient() as client:
        with pytest.raises(HTTPException) as forwarded:
            await forward(request, identity, "datasets", None, client, 1000)
    assert forwarded.value.status_code == 503


@pytest.mark.asyncio
async def test_forwarding_requires_a_user_or_authenticated_key(connection: Connection, identity: Identity) -> None:
    anonymous: Final = identity.model_copy(update={"user_id": None, "token": None})
    request: Final = Request({"type": "http", "method": "GET", "headers": []})
    async with httpx.AsyncClient() as client:
        with pytest.raises(HTTPException) as error:
            await forward(request, anonymous, "datasets", connection, client, 1000)
    assert error.value.status_code == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("malformed", "oversized", "transport"))
async def test_failed_status_response_never_reports_a_ready_service(connection: Connection, failure: str) -> None:
    def transport(request: httpx.Request) -> httpx.Response:
        if failure == "transport":
            raise httpx.ConnectError("Fixture connection refused", request=request)
        return httpx.Response(200, content=b"x" * (16 * 1024 + 1) if failure == "oversized" else b"invalid-json")

    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        result: Final = await service_status(connection, client)
    assert not result.storage_ready
    assert not result.credentials_ready
    assert result.public_contract == 0


@pytest.mark.asyncio
async def test_failed_forward_reports_service_unavailability(connection: Connection, identity: Identity) -> None:
    async def receive() -> Mapping[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    def transport(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Fixture response unavailable", request=request)

    request: Final = Request(
        {"type": "http", "method": "GET", "path": "/lens", "query_string": b"", "headers": []}, receive
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        with pytest.raises(HTTPException) as error:
            await forward(request, identity, "", connection, client, 1000)
    assert error.value.status_code == 503
    assert error.value.detail == "Lens service is unavailable"


SIGN_IN_URL: Final = "http://gateway.test/ui/login?redirect_to=%2Flens%2Flogin"


def test_lens_login_redirects_a_signed_in_user_to_lens_with_a_sign_in_only_token(
    connection: Connection, identity: Identity
) -> None:
    response: Final = login_redirect(connection, "https://lens.test", identity, None, SIGN_IN_URL, 1000)
    assert response.status_code == 303
    location: Final = response.headers["location"]
    prefix: Final = "https://lens.test/ui/#token="
    assert location.startswith(prefix)
    decoded: Final = jwt.decode(
        location[len(prefix) :],
        SECRET,
        algorithms=["HS256"],
        audience="litellm-lens-login",
        issuer="litellm",
        options={"verify_exp": False},
    )
    assert decoded == {
        "iss": "litellm",
        "aud": "litellm-lens-login",
        "sub": "user",
        "iat": 1000,
        "exp": 1060,
        "identity": identity.model_dump(mode="json"),
    }
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_lens_login_token_names_the_signed_in_user_and_request_tokens_never_do(
    connection: Connection, identity: Identity
) -> None:
    named: Final = connection.login_token(identity, "kerry@example.com", 1000)
    forwarded: Final = connection.identity_token(identity, 1000)
    assert named is not None
    assert forwarded is not None
    options: Final = {"verify_exp": False}
    assert jwt.decode(named, SECRET, algorithms=["HS256"], audience="litellm-lens-login", options=options)[
        "identity"
    ] == identity.model_dump(mode="json") | {"user_name": "kerry@example.com"}
    assert jwt.decode(forwarded, SECRET, algorithms=["HS256"], audience="litellm-lens", options=options)[
        "identity"
    ] == identity.model_dump(mode="json")


@pytest.mark.parametrize(
    ("user_id", "user_email", "expected"),
    (
        ("dev-user", "dev@example.com", "dev@example.com"),
        ("default_user_id", None, "root"),
        ("default_user_id", "owner@example.com", "owner@example.com"),
        ("dev-user", None, None),
    ),
    ids=("user_email", "built_in_admin_username", "admin_email_wins", "no_name_known"),
)
def test_lens_login_names_a_user_by_email_and_the_built_in_admin_by_its_username(
    user_id: str, user_email: str | None, expected: str | None
) -> None:
    assert login_user_name(UserAPIKeyAuth(user_id=user_id, user_email=user_email), "root") == expected


def test_lens_login_token_cannot_be_used_as_a_forwarded_request_identity(
    connection: Connection, identity: Identity
) -> None:
    token: Final = connection.login_token(identity, None, 1000)
    assert token is not None
    with pytest.raises(jwt.InvalidAudienceError):
        jwt.decode(token, SECRET, algorithms=["HS256"], audience="litellm-lens", options={"verify_exp": False})


def test_lens_login_sends_a_browser_without_a_dashboard_session_to_gateway_sign_in(connection: Connection) -> None:
    response: Final = login_redirect(connection, "https://lens.test", None, None, SIGN_IN_URL, 1000)
    assert (response.status_code, response.headers["location"]) == (303, SIGN_IN_URL)


@pytest.mark.parametrize(
    ("sign_in_url", "ui_cookie_path"),
    (
        ("http://gateway.test/ui/login?redirect_to=%2Flens%2Flogin", "/ui"),
        ("http://gateway.test/litellm/ui/login?redirect_to=%2Flitellm%2Flens%2Flogin", "/litellm/ui"),
    ),
    ids=("no_root_path", "server_root_path"),
)
def test_lens_login_drops_a_refused_dashboard_session_so_the_login_page_does_not_send_it_back(
    connection: Connection, sign_in_url: str, ui_cookie_path: str
) -> None:
    response: Final = login_redirect(connection, "https://lens.test", None, None, sign_in_url, 1000)
    cleared: Final = sorted(header for header in response.headers.getlist("set-cookie"))
    assert [
        (header.split(";")[0], "Max-Age=0" in header, f"Path={path}" in header.split("; "))
        for header, path in zip(cleared, sorted(("/", ui_cookie_path)), strict=True)
    ] == [
        ('token=""', True, True),
        ('token=""', True, True),
    ]


@pytest.mark.parametrize(
    ("configured", "lens_url"),
    ((False, "https://lens.test"), (True, "")),
    ids=("no_lens_connection", "no_public_lens_url"),
)
def test_lens_login_is_unavailable_until_lens_is_fully_configured(
    connection: Connection, identity: Identity, configured: bool, lens_url: str
) -> None:
    with pytest.raises(HTTPException) as error:
        login_redirect(connection if configured else None, lens_url, identity, None, SIGN_IN_URL, 1000)
    assert error.value.status_code == 503


def test_lens_login_refuses_an_identity_with_no_user_or_key(connection: Connection, identity: Identity) -> None:
    anonymous: Final = identity.model_copy(update={"user_id": None, "token": None})
    with pytest.raises(HTTPException) as error:
        login_redirect(connection, "https://lens.test", anonymous, None, SIGN_IN_URL, 1000)
    assert error.value.status_code == 403


def test_lens_login_sends_a_session_that_must_change_its_password_to_the_password_page() -> None:
    restricted: Final = UserAPIKeyAuth(
        user_id="user-1",
        allowed_routes=["/user/password/change", "/session/logout"],
        metadata={"password_reset_required": True},
    )
    response: Final = password_change_redirect(restricted, "/lens/login", "http://gateway.test")
    assert response is not None
    assert response.status_code == 303
    assert response.headers["location"] == "http://gateway.test/ui/change-password"


def test_lens_login_refuses_a_session_whose_key_may_not_call_it() -> None:
    restricted: Final = UserAPIKeyAuth(user_id="user-1", allowed_routes=["/key/info"])
    with pytest.raises(HTTPException) as error:
        password_change_redirect(restricted, "/lens/login", "http://gateway.test")
    assert error.value.status_code == 403


def test_lens_login_lets_an_unrestricted_session_and_a_missing_session_through() -> None:
    assert password_change_redirect(UserAPIKeyAuth(user_id="user-1"), "/lens/login", "http://gateway.test") is None
    assert password_change_redirect(None, "/lens/login", "http://gateway.test") is None


def test_lens_login_honours_admin_only_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "general_settings", {"admin_only_routes": ["/lens/login"]})
    monkeypatch.setattr(proxy_server, "premium_user", True)
    member: Final = UserAPIKeyAuth(user_id="user-1", user_role=LitellmUserRoles.INTERNAL_USER)
    admin: Final = UserAPIKeyAuth(user_id="admin-1", user_role=LitellmUserRoles.PROXY_ADMIN)
    with pytest.raises(HTTPException) as error:
        password_change_redirect(member, "/lens/login", "http://gateway.test")
    assert error.value.status_code == 403
    assert password_change_redirect(admin, "/lens/login", "http://gateway.test") is None
