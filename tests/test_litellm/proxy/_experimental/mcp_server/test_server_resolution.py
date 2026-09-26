from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Final, Literal
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from litellm.proxy._experimental.mcp_server.server_resolution import (
    ResolvedMCPServer,
    authorize_mcp_server,
    resolve_mcp_server,
)
from litellm.proxy._experimental.mcp_server.ui_session_utils import can_access_mcp_server
from litellm.proxy._types import LiteLLM_MCPServerTable, LitellmUserRoles, UserAPIKeyAuth
from litellm.types.mcp import MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer


@dataclass(frozen=True)
class FakeMCPServerManager:
    servers_by_id: Mapping[str, MCPServer]
    servers_by_name: Mapping[str, MCPServer]
    allowed_server_ids: tuple[str, ...]
    id_lookup_spy: Mock
    name_lookup_spy: Mock
    ip_filter_spy: Mock
    allowed_servers_spy: Mock
    ip_accessible: bool

    def get_mcp_server_by_id(self, server_id: str) -> MCPServer | None:
        self.id_lookup_spy(server_id)
        return self.servers_by_id.get(server_id)

    def get_mcp_server_by_name(self, server_name: str, client_ip: str | None = None) -> MCPServer | None:
        self.name_lookup_spy(server_name, client_ip)
        return self.servers_by_name.get(server_name)

    def _is_server_accessible_from_ip(self, server: MCPServer, client_ip: str) -> bool:
        self.ip_filter_spy(server, client_ip)
        return self.ip_accessible

    def _build_mcp_server_table(self, server: MCPServer) -> LiteLLM_MCPServerTable:
        return LiteLLM_MCPServerTable(
            server_id=server.server_id,
            alias=server.alias,
            server_name=server.server_name,
            url=server.url,
            transport=server.transport,
        )

    async def get_allowed_mcp_servers(self, user_api_key_auth: UserAPIKeyAuth) -> list[str]:
        self.allowed_servers_spy(user_api_key_auth)
        return list(self.allowed_server_ids)


def _runtime_server(server_id: str = "canonical-server") -> MCPServer:
    return MCPServer(
        server_id=server_id,
        name=server_id,
        alias=f"{server_id}-alias",
        server_name=f"{server_id}-name",
        url="https://example.com/mcp",
        transport=MCPTransport.http,
    )


def _table_server(server_id: str = "database-server") -> LiteLLM_MCPServerTable:
    return LiteLLM_MCPServerTable(
        server_id=server_id,
        alias=f"{server_id}-alias",
        server_name=f"{server_id}-name",
        url="https://example.com/mcp",
        transport=MCPTransport.http,
    )


def _manager(
    *,
    servers_by_id: Mapping[str, MCPServer] | None = None,
    servers_by_name: Mapping[str, MCPServer] | None = None,
    allowed_server_ids: tuple[str, ...] = (),
    ip_accessible: bool = True,
) -> FakeMCPServerManager:
    return FakeMCPServerManager(
        servers_by_id={} if servers_by_id is None else servers_by_id,
        servers_by_name={} if servers_by_name is None else servers_by_name,
        allowed_server_ids=allowed_server_ids,
        id_lookup_spy=Mock(),
        name_lookup_spy=Mock(),
        ip_filter_spy=Mock(),
        allowed_servers_spy=Mock(),
        ip_accessible=ip_accessible,
    )


def _auth() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        user_role=LitellmUserRoles.INTERNAL_USER,
        user_id="resolver-test-user",
        api_key="resolver-test-key",
    )


@pytest.mark.asyncio
async def test_temp_resolution_precedes_db_and_registry() -> None:
    temporary_server: Final = _runtime_server("temporary-server")
    manager: Final = _manager(servers_by_id={temporary_server.server_id: temporary_server})
    temp_lookup: Final[Mock] = Mock()
    db_lookup: Final[Mock] = Mock()

    async def lookup_temp(server_id: str) -> MCPServer | None:
        temp_lookup(server_id)
        return temporary_server

    async def lookup_db(server_id: str) -> LiteLLM_MCPServerTable | None:
        db_lookup(server_id)
        return _table_server(server_id)

    resolved: Final = await resolve_mcp_server(
        "requested-id",
        manager=manager,
        temp_lookup=lookup_temp,
        db_lookup=lookup_db,
    )

    assert resolved == ResolvedMCPServer(
        table=manager._build_mcp_server_table(temporary_server),
        runtime=temporary_server,
        source="temp",
    )
    temp_lookup.assert_called_once_with("requested-id")
    db_lookup.assert_not_called()
    manager.id_lookup_spy.assert_not_called()


@pytest.mark.asyncio
async def test_db_resolution_precedes_registry_id() -> None:
    database_server: Final = _table_server("database-server")
    registry_server: Final = _runtime_server(database_server.server_id)
    manager: Final = _manager(servers_by_id={registry_server.server_id: registry_server})
    db_lookup: Final = Mock()

    async def lookup_db(server_id: str) -> LiteLLM_MCPServerTable | None:
        db_lookup(server_id)
        return database_server

    resolved: Final = await resolve_mcp_server(
        database_server.server_id,
        manager=manager,
        db_lookup=lookup_db,
    )

    assert resolved == ResolvedMCPServer(table=database_server, runtime=None, source="db")
    db_lookup.assert_called_once_with(database_server.server_id)
    manager.id_lookup_spy.assert_not_called()


@pytest.mark.asyncio
async def test_registry_id_resolution_precedes_name() -> None:
    server: Final = _runtime_server()
    name_collision: Final = _runtime_server("other-server")
    manager: Final = _manager(
        servers_by_id={server.server_id: server},
        servers_by_name={server.server_id: name_collision},
    )

    resolved: Final = await resolve_mcp_server(
        server.server_id,
        manager=manager,
        match_name=True,
    )

    assert resolved == ResolvedMCPServer(
        table=manager._build_mcp_server_table(server),
        runtime=server,
        source="registry",
    )
    manager.id_lookup_spy.assert_called_once_with(server.server_id)
    manager.name_lookup_spy.assert_not_called()


@pytest.mark.asyncio
async def test_lookup_ip_arguments_are_scoped_and_name_matching_can_be_disabled() -> None:
    server: Final = _runtime_server()
    manager: Final = _manager(servers_by_name={"server-alias": server})

    resolved: Final = await resolve_mcp_server(
        "server-alias",
        manager=manager,
        id_client_ip="id-client",
        name_client_ip="name-client",
        match_name=True,
    )

    assert resolved is not None
    assert resolved.source == "registry"
    assert resolved.runtime == server
    manager.id_lookup_spy.assert_called_once_with("server-alias")
    manager.ip_filter_spy.assert_not_called()
    manager.name_lookup_spy.assert_called_once_with("server-alias", "name-client")

    disabled_manager: Final = _manager(servers_by_name={"server-alias": server})
    not_resolved: Final = await resolve_mcp_server(
        "server-alias",
        manager=disabled_manager,
        name_client_ip="name-client",
    )

    assert not_resolved is None
    disabled_manager.id_lookup_spy.assert_called_once_with("server-alias")
    disabled_manager.name_lookup_spy.assert_not_called()


@pytest.mark.asyncio
async def test_id_lookup_applies_ip_filter_after_unfiltered_registry_lookup() -> None:
    server: Final = _runtime_server()
    manager: Final = _manager(servers_by_id={server.server_id: server}, ip_accessible=False)

    resolved: Final = await resolve_mcp_server(
        server.server_id,
        manager=manager,
        id_client_ip="external-client",
    )

    assert resolved is None
    manager.id_lookup_spy.assert_called_once_with(server.server_id)
    manager.ip_filter_spy.assert_called_once_with(server, "external-client")
    manager.name_lookup_spy.assert_not_called()


@pytest.mark.asyncio
async def test_db_lookup_none_skips_db_and_returns_registry_source() -> None:
    server: Final = _runtime_server()
    manager: Final = _manager(servers_by_id={server.server_id: server})

    resolved: Final = await resolve_mcp_server(server.server_id, manager=manager, db_lookup=None)

    assert resolved == ResolvedMCPServer(
        table=manager._build_mcp_server_table(server),
        runtime=server,
        source="registry",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "is_admin_view,missing_policy,expected_status",
    [
        pytest.param(True, "not_found", 404, id="admin-view-not-found"),
        pytest.param(False, "not_found", 404, id="non-admin-not-found"),
        pytest.param(False, "forbidden", 403, id="non-admin-forbidden"),
    ],
)
async def test_authorize_missing_uses_caller_policy(
    is_admin_view: bool,
    missing_policy: Literal["not_found", "forbidden"],
    expected_status: int,
) -> None:
    manager: Final = _manager()
    with pytest.raises(HTTPException) as exc_info:
        await authorize_mcp_server(
            None,
            _auth(),
            manager=manager,
            is_admin_view=is_admin_view,
            not_found_detail={"error": "not found"},
            forbidden_detail={"error": "forbidden"},
            non_admin_missing=missing_policy,
        )

    assert exc_info.value.status_code == expected_status
    assert exc_info.value.detail == ({"error": "not found"} if expected_status == 404 else {"error": "forbidden"})
    manager.allowed_servers_spy.assert_not_called()


@pytest.mark.asyncio
async def test_non_admin_temp_resolution_is_denied_before_allowed_lookup() -> None:
    server: Final = _runtime_server()
    manager: Final = _manager(allowed_server_ids=(server.server_id,))
    resolved: Final = ResolvedMCPServer(
        table=manager._build_mcp_server_table(server),
        runtime=server,
        source="temp",
    )

    with pytest.raises(HTTPException) as exc_info:
        await authorize_mcp_server(
            resolved,
            _auth(),
            manager=manager,
            is_admin_view=False,
            not_found_detail={"error": "not found"},
            forbidden_detail={"error": "forbidden"},
            non_admin_missing="not_found",
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == {"error": "forbidden"}
    manager.allowed_servers_spy.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "allowed_server_ids,expected_status",
    [
        pytest.param(("canonical-server",), None, id="allowed-canonical-id"),
        pytest.param((), 403, id="denied-canonical-id"),
    ],
)
async def test_authorize_uses_real_access_helper_for_canonical_id(
    allowed_server_ids: tuple[str, ...],
    expected_status: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server: Final = _runtime_server("canonical-server")
    manager: Final = _manager(
        servers_by_name={"display-alias": server},
        allowed_server_ids=allowed_server_ids,
    )
    resolved: Final = await resolve_mcp_server(
        "display-alias",
        manager=manager,
        match_name=True,
    )
    assert resolved is not None
    access_spy: Final = Mock()

    async def spy_access(
        user_api_key_auth: UserAPIKeyAuth,
        requested_server_id: str,
        allowed_servers: Callable[[UserAPIKeyAuth], Awaitable[list[str]]],
    ) -> bool:
        access_spy(requested_server_id)
        return await can_access_mcp_server(user_api_key_auth, requested_server_id, allowed_servers)

    monkeypatch.setattr(
        "litellm.proxy._experimental.mcp_server.server_resolution.can_access_mcp_server",
        spy_access,
    )
    if expected_status is None:
        authorized: Final = await authorize_mcp_server(
            resolved,
            _auth(),
            manager=manager,
            is_admin_view=False,
            not_found_detail={"error": "not found"},
            forbidden_detail={"error": "forbidden"},
            non_admin_missing="not_found",
        )
        assert authorized is resolved
    else:
        with pytest.raises(HTTPException) as exc_info:
            await authorize_mcp_server(
                resolved,
                _auth(),
                manager=manager,
                is_admin_view=False,
                not_found_detail={"error": "not found"},
                forbidden_detail={"error": "forbidden"},
                non_admin_missing="not_found",
            )

        assert exc_info.value.status_code == expected_status
        assert exc_info.value.detail == {"error": "forbidden"}

    access_spy.assert_called_once_with(server.server_id)
    manager.allowed_servers_spy.assert_called_once()
