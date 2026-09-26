from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from fastapi import HTTPException, status

from litellm.proxy._experimental.mcp_server.ui_session_utils import can_access_mcp_server
from litellm.proxy._types import LiteLLM_MCPServerTable, UserAPIKeyAuth
from litellm.types.mcp_server.mcp_server_manager import MCPServer

if TYPE_CHECKING:
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager

ResolutionSource = Literal["temp", "db", "registry"]


@dataclass(frozen=True)
class ResolvedMCPServer:
    table: LiteLLM_MCPServerTable
    runtime: MCPServer | None
    source: ResolutionSource


async def resolve_mcp_server(
    server_id: str,
    *,
    manager: MCPServerManager,
    db_lookup: Callable[[str], Awaitable[LiteLLM_MCPServerTable | None]] | None = None,
    temp_lookup: Callable[[str], Awaitable[MCPServer | None]] | None = None,
    id_client_ip: str | None = None,
    name_client_ip: str | None = None,
    match_name: bool = False,
) -> ResolvedMCPServer | None:
    if temp_lookup is not None:
        temporary_server: MCPServer | None = await temp_lookup(server_id)
        if temporary_server is not None:
            return ResolvedMCPServer(
                table=manager._build_mcp_server_table(temporary_server),
                runtime=temporary_server,
                source="temp",
            )

    if db_lookup is not None:
        database_server: LiteLLM_MCPServerTable | None = await db_lookup(server_id)
        if database_server is not None:
            return ResolvedMCPServer(table=database_server, runtime=None, source="db")

    registry_candidate: Final[MCPServer | None] = manager.get_mcp_server_by_id(server_id)
    registry_server: Final[MCPServer | None] = (
        registry_candidate
        if registry_candidate is not None
        and (id_client_ip is None or manager._is_server_accessible_from_ip(registry_candidate, id_client_ip))
        else None
    )
    if registry_server is not None:
        return ResolvedMCPServer(
            table=manager._build_mcp_server_table(registry_server),
            runtime=registry_server,
            source="registry",
        )

    if match_name:
        named_server: MCPServer | None = manager.get_mcp_server_by_name(server_id, client_ip=name_client_ip)
        if named_server is not None:
            return ResolvedMCPServer(
                table=manager._build_mcp_server_table(named_server),
                runtime=named_server,
                source="registry",
            )

    return None


async def authorize_mcp_server(
    resolved: ResolvedMCPServer | None,
    user_api_key_dict: UserAPIKeyAuth,
    *,
    manager: MCPServerManager,
    is_admin_view: bool,
    not_found_detail: Mapping[str, str],
    forbidden_detail: Mapping[str, str],
    non_admin_missing: Literal["not_found", "forbidden"],
) -> ResolvedMCPServer:
    if resolved is None:
        if is_admin_view or non_admin_missing == "not_found":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=dict(not_found_detail),
            )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=dict(forbidden_detail),
        )

    if is_admin_view:
        return resolved

    if resolved.source == "temp" or not await can_access_mcp_server(
        user_api_key_dict,
        resolved.table.server_id,
        manager.get_allowed_mcp_servers,
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=dict(forbidden_detail),
        )

    return resolved
