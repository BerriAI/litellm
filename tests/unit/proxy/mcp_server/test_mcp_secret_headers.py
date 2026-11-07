import io
import logging
from collections.abc import Mapping
from typing import Final

import httpx2
import pytest
from starlette.datastructures import Headers

from litellm._logging import verbose_logger
from litellm.mcp_client.client import MCPClient
from litellm.proxy._types import LiteLLM_MCPServerTable, NewMCPServerRequest
from litellm.proxy.mcp_server import rest_endpoints
from litellm.proxy.mcp_server.db import encrypt_credentials
from litellm.proxy.mcp_server.mcp_server_manager import MCPServerManager, global_mcp_server_manager
from litellm.types.mcp import MCPAuth, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer

_URL: Final = "https://upstream.example/mcp"
_CREDENTIAL_HEADERS: Final = {"DD-API-KEY": "dd-api-secret", "DD-APPLICATION-KEY": "dd-app-secret"}
_STATIC_HEADERS: Final = {"X-Datadog-Site": "datadoghq.com"}
_CREDENTIAL_HEADER_TEXT: Final = (*_CREDENTIAL_HEADERS, *_CREDENTIAL_HEADERS.values())


@pytest.fixture(autouse=True)
def _salt_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt-key-for-secret-header-tests-1234")


def _table(*, encrypted: bool) -> LiteLLM_MCPServerTable:
    credentials: Final = {"headers": dict(_CREDENTIAL_HEADERS)}
    return LiteLLM_MCPServerTable(
        server_id="srv-dd",
        server_name="dd",
        url=_URL,
        transport=MCPTransport.http,
        auth_type=MCPAuth.secret_headers,
        credentials=encrypt_credentials(credentials, encryption_key=None) if encrypted else credentials,
        static_headers=dict(_STATIC_HEADERS),
    )


async def _server(*, encrypted: bool) -> MCPServer:
    return await MCPServerManager().build_mcp_server_from_table(
        _table(encrypted=encrypted), credentials_are_encrypted=encrypted
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("encrypted", [True, False])
async def test_secret_header_server_sends_credential_and_static_headers_upstream(encrypted: bool) -> None:
    manager: Final = MCPServerManager()
    server: Final = await _server(encrypted=encrypted)
    extra_headers: Final = await manager._resolve_static_headers_with_env_vars(server, None)
    client: Final = await manager._create_mcp_client(server, extra_headers=extra_headers)
    sent: Final = (await client.prepare_request_auth()).headers
    expected: Final = {**_STATIC_HEADERS, **_CREDENTIAL_HEADERS}
    assert {name: sent[name] for name in expected} == expected


@pytest.mark.asyncio
async def test_admin_table_of_secret_header_server_lists_only_non_secret_static_headers() -> None:
    server: Final = await _server(encrypted=True)
    table: Final = MCPServerManager().build_mcp_server_table(server)
    assert table.static_headers == _STATIC_HEADERS
    assert table.credentials is None


@pytest.mark.asyncio
async def test_instructions_prefetch_authenticates_secret_header_server_with_its_headers() -> None:
    sent: Final[list[httpx2.Headers]] = []  # mutable-ok: records upstream requests

    def upstream(request: httpx2.Request) -> httpx2.Response:
        sent.append(request.headers)
        return httpx2.Response(500)

    manager: Final = MCPServerManager(upstream_transport=httpx2.MockTransport(upstream))
    await manager.ensure_upstream_initialize_instructions_cached(await _server(encrypted=True))
    expected: Final = {**_STATIC_HEADERS, **_CREDENTIAL_HEADERS}
    assert {name: sent[0][name] for name in expected} == expected


@pytest.mark.asyncio
async def test_health_check_authenticates_secret_header_server_with_its_headers() -> None:
    sent: Final[list[httpx2.Headers]] = []  # mutable-ok: records upstream requests

    def upstream(request: httpx2.Request) -> httpx2.Response:
        sent.append(request.headers)
        return httpx2.Response(500)

    manager: Final = MCPServerManager(upstream_transport=httpx2.MockTransport(upstream))
    server: Final = await _server(encrypted=True)
    manager.registry[server.server_id] = server
    result: Final = await manager.health_check_server(server.server_id)
    expected: Final = {**_STATIC_HEADERS, **_CREDENTIAL_HEADERS}
    assert {name: sent[0][name] for name in expected} == expected
    assert result.static_headers == _STATIC_HEADERS


def test_secret_header_server_advertises_the_gateway_authorization_server_like_an_api_key_server() -> None:
    def advertises(auth_type: MCPAuth) -> bool:
        return MCPServer(
            server_id="s", name="s", url=_URL, transport=MCPTransport.http, auth_type=auth_type
        ).advertises_gateway_authorization_server

    assert advertises(MCPAuth.secret_headers) is advertises(MCPAuth.api_key) is True


def test_preview_sends_the_credential_headers_with_the_static_headers() -> None:
    request: Final = NewMCPServerRequest(
        url=_URL,
        auth_type=MCPAuth.secret_headers,
        credentials={"headers": dict(_CREDENTIAL_HEADERS)},
        static_headers=dict(_STATIC_HEADERS),
    )
    assert rest_endpoints._preview_static_headers(request) == {**_STATIC_HEADERS, **_CREDENTIAL_HEADERS}


@pytest.mark.asyncio
async def test_credential_headers_are_ignored_unless_the_auth_type_is_secret_headers() -> None:
    table: Final = _table(encrypted=True).model_copy(update={"auth_type": MCPAuth.api_key})
    server: Final = await MCPServerManager().build_mcp_server_from_table(table, credentials_are_encrypted=True)
    assert server.static_headers == _STATIC_HEADERS
    assert server.credential_header_names == ()


@pytest.mark.asyncio
async def test_a_credential_header_replaces_a_static_header_of_the_same_name_in_any_case() -> None:
    table: Final = _table(encrypted=True).model_copy(
        update={"static_headers": {**_STATIC_HEADERS, "dd-api-key": "stale-plaintext"}}
    )
    manager: Final = MCPServerManager()
    server: Final = await manager.build_mcp_server_from_table(table, credentials_are_encrypted=True)
    assert server.static_headers == {**_STATIC_HEADERS, **_CREDENTIAL_HEADERS}
    assert manager.build_mcp_server_table(server).static_headers == _STATIC_HEADERS


@pytest.mark.asyncio
async def test_env_var_placeholders_are_filled_in_static_headers_but_not_in_secret_values() -> None:
    server: Final = MCPServer(
        server_id="srv-env",
        name="srv-env",
        url=_URL,
        transport=MCPTransport.http,
        auth_type=MCPAuth.secret_headers,
        static_headers={"X-Site": "${SITE}", "DD-API-KEY": "key-${SITE}"},
        credential_header_names=("DD-API-KEY",),
        env_vars=[{"name": "SITE", "value": "datadoghq.com", "scope": "global"}],
    )
    resolved: Final = await MCPServerManager()._resolve_static_headers_with_env_vars(server, None)
    assert resolved == {"X-Site": "datadoghq.com", "DD-API-KEY": "key-${SITE}"}


@pytest.mark.asyncio
@pytest.mark.parametrize("caller_token", [None, "caller-token"])
@pytest.mark.parametrize(
    "redirect_to,expected",
    [
        ("https://upstream.example/other", {"DD-API-KEY": "dd-api-secret", "DD-APPLICATION-KEY": "dd-app-secret"}),
        ("https://elsewhere.example/collect", {"DD-API-KEY": None, "DD-APPLICATION-KEY": None}),
    ],
)
async def test_credential_headers_do_not_follow_a_redirect_to_another_origin(
    redirect_to: str, expected: Mapping[str, str | None], caller_token: str | None
) -> None:
    seen: Final[list[httpx2.Headers]] = []  # mutable-ok: records upstream requests

    def upstream(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/mcp":
            return httpx2.Response(307, headers={"location": redirect_to})
        seen.append(request.headers)
        return httpx2.Response(200)

    manager: Final = MCPServerManager()
    server: Final = await _server(encrypted=True)
    client: Final = await manager._create_mcp_client(
        server,
        mcp_auth_header=caller_token,
        extra_headers=await manager._resolve_static_headers_with_env_vars(server, None),
    )
    async with client._create_httpx_client_factory(transport=httpx2.MockTransport(upstream))(
        headers=client._get_auth_headers()
    ) as http:
        await http.get(_URL)
    assert {name: seen[0].get(name) for name in expected} == expected


@pytest.mark.parametrize(
    "preview_url,headers",
    [(_URL, _CREDENTIAL_HEADERS), ("https://elsewhere.example/mcp", {})],
)
@pytest.mark.asyncio
async def test_preview_of_a_saved_server_reuses_its_headers_only_for_the_same_origin(
    monkeypatch: pytest.MonkeyPatch, preview_url: str, headers: Mapping[str, str]
) -> None:
    saved: Final = await _server(encrypted=True)
    monkeypatch.setitem(global_mcp_server_manager.registry, saved.server_id, saved)
    request: Final = NewMCPServerRequest(
        server_id=saved.server_id, url=preview_url, transport=MCPTransport.http, auth_type=MCPAuth.secret_headers
    )
    staged: Final = rest_endpoints._stage_server_test(request, Headers({}))
    assert rest_endpoints._preview_static_headers(staged.request) == (headers or None)


@pytest.mark.asyncio
async def test_debug_log_of_the_upstream_headers_leaves_out_credential_headers() -> None:
    manager: Final = MCPServerManager()
    server: Final = await _server(encrypted=True)
    client: Final = await manager._create_mcp_client(
        server, extra_headers=await manager._resolve_static_headers_with_env_vars(server, None)
    )
    captured: Final = io.StringIO()
    handler: Final = logging.StreamHandler(captured)
    previous_level: Final = verbose_logger.level
    verbose_logger.addHandler(handler)
    verbose_logger.setLevel(logging.DEBUG)
    try:
        client._create_transport_context()
    finally:
        verbose_logger.removeHandler(handler)
        verbose_logger.setLevel(previous_level)
    logged: Final = captured.getvalue()
    assert all(text not in logged for text in _CREDENTIAL_HEADER_TEXT)
    assert _STATIC_HEADERS["X-Datadog-Site"] in logged


@pytest.mark.asyncio
async def test_preview_client_keeps_credential_headers_out_of_the_debug_log() -> None:
    request: Final = NewMCPServerRequest(
        url=_URL,
        transport=MCPTransport.http,
        auth_type=MCPAuth.secret_headers,
        credentials={"headers": dict(_CREDENTIAL_HEADERS)},
        static_headers=dict(_STATIC_HEADERS),
    )

    async def debug_log_of_transport(client: MCPClient) -> Mapping[str, object]:
        captured: Final = io.StringIO()
        handler: Final = logging.StreamHandler(captured)
        previous_level: Final = verbose_logger.level
        verbose_logger.addHandler(handler)
        verbose_logger.setLevel(logging.DEBUG)
        try:
            client._create_transport_context()
        finally:
            verbose_logger.removeHandler(handler)
            verbose_logger.setLevel(previous_level)
        return {"log": captured.getvalue()}

    result: Final = await rest_endpoints._execute_with_mcp_client(request, debug_log_of_transport)
    logged: Final = str(result["log"])
    assert all(text not in logged for text in _CREDENTIAL_HEADER_TEXT)
    assert _STATIC_HEADERS["X-Datadog-Site"] in logged


@pytest.mark.parametrize("auth_type", [MCPAuth.none, MCPAuth.api_key])
def test_preview_ignores_credential_headers_for_any_other_auth_type(auth_type: MCPAuth) -> None:
    request: Final = NewMCPServerRequest(
        url=_URL,
        auth_type=auth_type,
        credentials={"headers": dict(_CREDENTIAL_HEADERS)},
        static_headers=dict(_STATIC_HEADERS),
    )
    assert rest_endpoints._preview_static_headers(request) == _STATIC_HEADERS
    assert rest_endpoints._preview_credential_header_names(request) == ()


@pytest.mark.asyncio
async def test_config_file_server_with_secret_headers_is_rejected_when_loaded() -> None:
    manager: Final = MCPServerManager()
    config: Final = {
        "datadog": {
            "url": _URL,
            "transport": MCPTransport.http,
            "auth_type": MCPAuth.secret_headers,
            "credentials": {"headers": dict(_CREDENTIAL_HEADERS)},
        }
    }
    with pytest.raises(ValueError, match="'datadog'.*secret_headers"):
        await manager.load_servers_from_config(config)
    assert not manager.config_mcp_servers


@pytest.mark.asyncio
@pytest.mark.parametrize("auth_type", [MCPAuth.none, MCPAuth.api_key])
async def test_config_file_server_with_another_auth_type_still_loads(auth_type: MCPAuth) -> None:
    manager: Final = MCPServerManager()
    config: Final = {
        "datadog": {
            "url": _URL,
            "transport": MCPTransport.http,
            "auth_type": auth_type,
            "static_headers": _STATIC_HEADERS,
        }
    }
    await manager.load_servers_from_config(config)
    assert [server.auth_type for server in manager.config_mcp_servers.values()] == [auth_type]
