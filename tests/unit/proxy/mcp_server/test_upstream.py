import json
import queue
from collections.abc import Callable, Iterator, Mapping
from typing import Final
from unittest.mock import MagicMock

import httpx2
import pytest
from mcp.server.context import ServerRequestContext
from mcp.types import CallToolResult

from litellm.proxy.mcp_server.mcp_context import active_mcp_request_ctx_var
from litellm.proxy.mcp_server.mcp_server_manager import MCPServerManager
from litellm.proxy.mcp_server.outbound_credentials.resolver import UpstreamCredentialProvider
from litellm.proxy.mcp_server.upstream import prepare_upstream_client
from litellm.types.mcp import MCPAuth, MCPTransport, MCPUpstreamProtocol
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from tests._support.mcp_count_tool_upstream import count_tool_upstream, methods_received

_HTTP_URL: Final = "https://upstream.example/mcp"
_SERVER_SHAPES: Final = pytest.mark.parametrize(
    ("server_fields", "per_request_token", "forwarded_token"),
    [
        ({"transport": MCPTransport.http, "url": _HTTP_URL}, None, None),
        (
            {
                "transport": MCPTransport.http,
                "url": _HTTP_URL,
                "auth_type": MCPAuth.bearer_token,
                "authentication_token": "static-token",
            },
            None,
            None,
        ),
        ({"transport": MCPTransport.http, "url": _HTTP_URL, "auth_type": MCPAuth.bearer_token}, "caller-key", None),
        ({"transport": MCPTransport.http, "url": _HTTP_URL, "auth_type": MCPAuth.oauth_delegate}, None, "caller-token"),
    ],
    ids=["no-auth", "static-bearer-token", "per-request-token", "forwarded-caller-token"],
)


@pytest.fixture
def _active_mcp_request_context() -> Iterator[None]:
    context: Final = ServerRequestContext(
        session=MagicMock(), lifespan_context={}, protocol_version="2025-11-25", method="tools/call", request=None
    )
    token: Final = active_mcp_request_ctx_var.set(context)
    try:
        yield
    finally:
        active_mcp_request_ctx_var.reset(token)


def _server(server_fields: Mapping[str, object], skip_setting: bool | None) -> MCPServer:
    return MCPServer.model_validate(
        {
            **server_fields,
            "server_id": "skip-validation",
            "name": "skip_validation",
            "mcp_info": {} if skip_setting is None else {"skip_tool_output_validation": skip_setting},
        }
    )


async def _call_count_tool(
    manager: MCPServerManager,
    server: MCPServer,
    *,
    per_request_token: str | None = None,
    forwarded_token: str | None = None,
    gateway_session_id: str | None = None,
) -> CallToolResult:
    result: Final = await manager._call_regular_mcp_tool(
        mcp_server=server,
        original_tool_name="count",
        arguments={},
        tasks=[],
        mcp_auth_header=per_request_token,
        mcp_server_auth_headers=None,
        oauth2_headers=None if forwarded_token is None else {"Authorization": f"Bearer {forwarded_token}"},
        raw_headers=None if gateway_session_id is None else {"mcp-session-id": gateway_session_id},
        proxy_logging_obj=None,
    )
    assert isinstance(result, CallToolResult)
    return result


@pytest.mark.asyncio
@_SERVER_SHAPES
@pytest.mark.parametrize(
    ("skip_setting", "rejected"), [(None, True), (False, True), (True, False)], ids=["unset", "off", "on"]
)
async def test_the_setting_alone_decides_whether_a_tool_result_is_checked(
    server_fields: Mapping[str, object],
    per_request_token: str | None,
    forwarded_token: str | None,
    skip_setting: bool | None,
    rejected: bool,
) -> None:
    upstream: Final = count_tool_upstream({"n": "not-an-integer"}, queue.SimpleQueue())
    manager: Final = MCPServerManager(upstream_transport=httpx2.MockTransport(upstream))

    result: Final = await _call_count_tool(
        manager,
        _server(server_fields, skip_setting),
        per_request_token=per_request_token,
        forwarded_token=forwarded_token,
    )

    assert result.is_error is rejected


@pytest.mark.asyncio
@_SERVER_SHAPES
async def test_a_skipping_server_is_never_asked_for_its_tools(
    server_fields: Mapping[str, object], per_request_token: str | None, forwarded_token: str | None
) -> None:
    methods_seen: Final[queue.SimpleQueue[str]] = queue.SimpleQueue()
    manager: Final = MCPServerManager(
        upstream_transport=httpx2.MockTransport(count_tool_upstream({"n": 1}, methods_seen))
    )

    result: Final = await _call_count_tool(
        manager,
        _server(server_fields, True),
        per_request_token=per_request_token,
        forwarded_token=forwarded_token,
    )

    assert result.is_error is False
    assert "tools/list" not in methods_received(methods_seen)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_setting", "second_setting", "rejected_in_order"),
    [(True, False, (False, True)), (False, True, (True, False))],
    ids=["turned-off", "turned-on"],
)
async def test_a_changed_setting_applies_to_the_next_call_in_a_live_gateway_session(
    _active_mcp_request_context: None,
    first_setting: bool,
    second_setting: bool,
    rejected_in_order: tuple[bool, bool],
) -> None:
    upstream: Final = count_tool_upstream({"n": "not-an-integer"}, queue.SimpleQueue())
    manager: Final = MCPServerManager(upstream_transport=httpx2.MockTransport(upstream))
    manager.track_gateway_session("gateway-1")
    fields: Final = {"transport": MCPTransport.http, "url": _HTTP_URL}
    try:
        first: Final = await _call_count_tool(manager, _server(fields, first_setting), gateway_session_id="gateway-1")
        second: Final = await _call_count_tool(manager, _server(fields, second_setting), gateway_session_id="gateway-1")
    finally:
        await manager.close_all_upstream_sessions()

    assert (first.is_error, second.is_error) == rejected_in_order


@pytest.mark.asyncio
async def test_a_stdio_upstream_client_carries_the_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_ENABLE_MCP_STDIO", "true")
    server: Final = _server({"transport": MCPTransport.stdio, "command": "python", "args": ["server.py"]}, True)

    client: Final = await prepare_upstream_client(
        server, provider=UpstreamCredentialProvider(), root_path="/mcp", protocol_version="auto"
    )

    assert client.skip_tool_output_validation is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("protocol_version", "expected"), [("auto", True), ("2025-11-25", True), ("2026-07-28", False)]
)
async def test_upstream_client_ignores_the_setting_where_the_call_already_lists_tools(
    protocol_version: MCPUpstreamProtocol, expected: bool
) -> None:
    server: Final = MCPServer.model_validate(
        {
            "server_id": "skip-validation",
            "name": "skip_validation",
            "url": _HTTP_URL,
            "transport": MCPTransport.http,
            "protocol_version": protocol_version,
            "mcp_info": {"skip_tool_output_validation": True},
        }
    )

    client: Final = await prepare_upstream_client(
        server, provider=UpstreamCredentialProvider(), root_path="/mcp", protocol_version=protocol_version
    )

    assert client.skip_tool_output_validation is expected


_SIGNED_URL: Final = "https://reports.example/mcp"


def _header_recording_upstream(
    authorizations: "queue.SimpleQueue[tuple[str, str | None]]",
) -> Callable[[httpx2.Request], httpx2.Response]:
    respond: Final = count_tool_upstream({"n": 1}, queue.SimpleQueue())

    def record(request: httpx2.Request) -> httpx2.Response:
        if request.method == "POST":
            authorizations.put((str(request.url), request.headers.get("authorization")))
        return respond(request)

    return record


@pytest.fixture
def _signing_key(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    from litellm.proxy.guardrails.guardrail_hooks.mcp_jwt_signer import mcp_jwt_signer

    pem: Final = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
        .decode()
    )
    monkeypatch.setenv("MCP_JWT_SIGNING_KEY", pem)
    monkeypatch.setattr(mcp_jwt_signer, "_mcp_jwt_signer_instance", None)
    yield


def _signed_server(url: str = _SIGNED_URL, server_id: str = "reports") -> MCPServer:
    return MCPServer.model_validate(
        {
            "server_id": server_id,
            "name": server_id,
            "transport": MCPTransport.http,
            "url": url,
            "auth_type": MCPAuth.signed_identity,
        }
    )


def _verified_claims(authorization: str | None, audience: str) -> Mapping[str, object]:
    import jwt
    from jwt.algorithms import RSAAlgorithm

    from litellm.proxy.guardrails.guardrail_hooks.mcp_jwt_signer.mcp_jwt_signer import get_published_signer

    signer: Final = get_published_signer()
    assert signer is not None and authorization is not None
    token: Final = authorization.removeprefix("Bearer ")
    jwk: Final = next(k for k in signer.get_jwks()["keys"] if k["kid"] == jwt.get_unverified_header(token)["kid"])
    public_key: Final = RSAAlgorithm.from_jwk(json.dumps(dict(jwk)))
    return jwt.decode(token, public_key, algorithms=["RS256"], audience=audience)


@pytest.mark.asyncio
@pytest.mark.usefixtures("_signing_key", "_active_mcp_request_context")
@pytest.mark.parametrize(
    ("caller", "expected_sub"),
    [
        ({"user_id": "u-1", "user_email": "maria@example.com"}, "maria@example.com"),
        ({"user_id": "u-2"}, "u-2"),
    ],
    ids=["email", "user-id-without-email"],
)
async def test_a_signed_identity_server_gets_the_caller_signed_by_the_published_key(
    caller: Mapping[str, str], expected_sub: str
) -> None:
    from litellm.proxy._types import UserAPIKeyAuth

    authorizations: Final[queue.SimpleQueue[tuple[str, str | None]]] = queue.SimpleQueue()
    manager: Final = MCPServerManager(
        upstream_transport=httpx2.MockTransport(_header_recording_upstream(authorizations))
    )

    result: Final = await manager._call_regular_mcp_tool(
        mcp_server=_signed_server(),
        original_tool_name="count",
        arguments={},
        tasks=[],
        mcp_auth_header="forged-by-client",
        mcp_server_auth_headers=None,
        oauth2_headers={"Authorization": "Bearer forwarded-by-client"},
        raw_headers=None,
        proxy_logging_obj=None,
        user_api_key_auth=UserAPIKeyAuth(**caller),
    )

    assert result.is_error is False
    seen: Final = [authorizations.get_nowait()[1] for _ in range(authorizations.qsize())]
    assert seen
    assert {_verified_claims(a, _SIGNED_URL)["sub"] for a in seen} == {expected_sub}


@pytest.mark.asyncio
@pytest.mark.usefixtures("_signing_key", "_active_mcp_request_context")
async def test_a_signed_token_for_one_server_is_rejected_by_another() -> None:
    import jwt

    from litellm.proxy._types import UserAPIKeyAuth

    authorizations: Final[queue.SimpleQueue[tuple[str, str | None]]] = queue.SimpleQueue()
    manager: Final = MCPServerManager(
        upstream_transport=httpx2.MockTransport(_header_recording_upstream(authorizations))
    )

    await manager._call_regular_mcp_tool(
        mcp_server=_signed_server(),
        original_tool_name="count",
        arguments={},
        tasks=[],
        mcp_auth_header=None,
        mcp_server_auth_headers=None,
        oauth2_headers=None,
        raw_headers=None,
        proxy_logging_obj=None,
        user_api_key_auth=UserAPIKeyAuth(user_id="u-1"),
    )

    _, authorization = authorizations.get_nowait()
    with pytest.raises(jwt.InvalidAudienceError):
        _verified_claims(authorization, "https://billing.example/mcp")


@pytest.mark.asyncio
@pytest.mark.usefixtures("_signing_key", "_active_mcp_request_context")
async def test_signing_for_a_signed_identity_server_leaves_other_servers_unsigned() -> None:
    from litellm.proxy._types import UserAPIKeyAuth

    authorizations: Final[queue.SimpleQueue[tuple[str, str | None]]] = queue.SimpleQueue()
    manager: Final = MCPServerManager(
        upstream_transport=httpx2.MockTransport(_header_recording_upstream(authorizations))
    )
    plain: Final = MCPServer.model_validate(
        {"server_id": "plain", "name": "plain", "transport": MCPTransport.http, "url": "https://plain.example/mcp"}
    )
    caller: Final = UserAPIKeyAuth(user_id="u-1")

    for server in (_signed_server(), plain):
        await manager._call_regular_mcp_tool(
            mcp_server=server,
            original_tool_name="count",
            arguments={},
            tasks=[],
            mcp_auth_header=None,
            mcp_server_auth_headers=None,
            oauth2_headers=None,
            raw_headers=None,
            proxy_logging_obj=None,
            user_api_key_auth=caller,
        )
    await manager.get_tools_from_server(server=plain, user_api_key_auth=caller, use_tool_list_cache=False)

    seen: Final = [authorizations.get_nowait() for _ in range(authorizations.qsize())]
    assert {auth for url, auth in seen if url.startswith("https://plain.example")} == {None}


@pytest.mark.asyncio
@pytest.mark.usefixtures("_active_mcp_request_context")
async def test_a_signed_identity_server_fails_closed_without_a_signing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from fastapi import HTTPException

    from litellm.proxy._types import UserAPIKeyAuth

    monkeypatch.delenv("MCP_JWT_SIGNING_KEY", raising=False)
    authorizations: Final[queue.SimpleQueue[tuple[str, str | None]]] = queue.SimpleQueue()
    manager: Final = MCPServerManager(
        upstream_transport=httpx2.MockTransport(_header_recording_upstream(authorizations))
    )

    with pytest.raises(HTTPException, match="MCP_JWT_SIGNING_KEY"):
        await manager._call_regular_mcp_tool(
            mcp_server=_signed_server(),
            original_tool_name="count",
            arguments={},
            tasks=[],
            mcp_auth_header=None,
            mcp_server_auth_headers=None,
            oauth2_headers=None,
            raw_headers=None,
            proxy_logging_obj=None,
            user_api_key_auth=UserAPIKeyAuth(user_id="u-1"),
        )
    assert authorizations.qsize() == 0


@pytest.mark.asyncio
@pytest.mark.usefixtures("_signing_key", "_active_mcp_request_context")
async def test_the_signer_guardrail_cannot_replace_a_signed_identity_servers_token() -> None:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.guardrails.guardrail_hooks.mcp_jwt_signer.mcp_jwt_signer import MCPJWTSigner

    MCPJWTSigner(guardrail_name="mcp-jwt-signer", audience="mcp")
    authorizations: Final[queue.SimpleQueue[tuple[str, str | None]]] = queue.SimpleQueue()
    manager: Final = MCPServerManager(
        upstream_transport=httpx2.MockTransport(_header_recording_upstream(authorizations))
    )

    await manager.get_tools_from_server(
        server=_signed_server(), user_api_key_auth=UserAPIKeyAuth(user_id="u-1"), use_tool_list_cache=False
    )

    seen: Final = [authorizations.get_nowait()[1] for _ in range(authorizations.qsize())]
    assert seen
    assert {_verified_claims(a, _SIGNED_URL)["aud"] for a in seen} == {_SIGNED_URL}


@pytest.mark.asyncio
@pytest.mark.usefixtures("_signing_key")
@pytest.mark.parametrize(
    ("byok_header", "per_server_headers"),
    [(None, None), ("Bearer forged-byok", None), (None, {"reports": {"Authorization": "Bearer forged-per-server"}})],
    ids=["no-client-header", "x-mcp-auth", "x-mcp-reports-authorization"],
)
async def test_an_openapi_signed_identity_server_sends_the_signed_caller_not_a_client_header(
    byok_header: str | None, per_server_headers: dict[str, dict[str, str]] | None
) -> None:
    from unittest.mock import patch

    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.mcp_server.openapi_to_mcp_generator import _merge_openapi_tool_request_headers

    server: Final = MCPServer.model_validate(
        {
            "server_id": "reports",
            "name": "reports",
            "server_name": "reports",
            "alias": "reports",
            "transport": MCPTransport.http,
            "url": _SIGNED_URL,
            "spec_path": "https://reports.example/openapi.json",
            "auth_type": MCPAuth.signed_identity,
        }
    )
    manager: Final = MCPServerManager()
    wire: dict[str, Mapping[str, str]] = {}  # mutable-ok: captures the headers the egress would send

    async def egress(_server: object, _name: object, _arguments: object, _wire_compat: object) -> CallToolResult:
        wire["headers"] = _merge_openapi_tool_request_headers({})
        return CallToolResult(content=[], isError=False)

    with (
        patch.object(manager, "_resolve_mcp_server_for_tool_call", return_value=server),
        patch.object(manager, "_call_openapi_tool_handler", side_effect=egress),
    ):
        await manager.call_tool(
            server_name="reports",
            name="whoami",
            arguments={},
            user_api_key_auth=UserAPIKeyAuth(user_id="u-1", user_email="maria@example.com"),
            mcp_auth_header=byok_header,
            mcp_server_auth_headers=per_server_headers,
        )

    authorizations: Final = [v for k, v in wire["headers"].items() if k.lower() == "authorization"]
    assert len(authorizations) == 1
    assert _verified_claims(authorizations[0], _SIGNED_URL)["sub"] == "maria@example.com"
