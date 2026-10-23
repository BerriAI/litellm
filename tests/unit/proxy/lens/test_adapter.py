from collections.abc import Mapping
from typing import Final

import httpx
import jwt
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.lens.adapter import (
    Connection,
    Identity,
    delegated_identity,
    forward,
    lens_request,
    request_body,
    service_connection,
    service_status,
)
from litellm.tracing.remote import LensConnection

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
