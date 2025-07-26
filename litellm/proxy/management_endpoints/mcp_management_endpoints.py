


"""
1. Allow proxy admin to perform create, update, and delete operations on MCP servers in the db.
2. Allows users to view the mcp servers they have access to.

Endpoints here:
- GET `/v1/mcp/server` - Returns all of the configured mcp servers in the db filtered by requestor's access
- GET `/v1/mcp/server/{server_id}` - Returns the the specific mcp server in the db given `server_id` filtered by requestor's access
- GET `/v1/mcp/server/{server_id}/tools` - Get all the tools from the mcp server specified by the `server_id`
- POST `/v1/mcp/server` - Add a new external mcp server.
- PUT `/v1/mcp/server` -  Edits an existing mcp server.
- DELETE `/v1/mcp/server/{server_id}` - Deletes the mcp server given `server_id`.
- GET `/v1/mcp/tools - lists all the tools available for a key
- GET `/v1/mcp/access_groups` - lists all available MCP access groups

"""

import importlib
from typing import Dict, Iterable, List, Optional, cast

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status
from fastapi.responses import JSONResponse

import litellm
from litellm._logging import verbose_logger, verbose_proxy_logger
from litellm.constants import LITELLM_PROXY_ADMIN_NAME
from litellm.proxy._experimental.mcp_server.utils import validate_and_normalize_mcp_server_payload

router = APIRouter(prefix="/v1/mcp", tags=["mcp"])
MCP_AVAILABLE: bool = True
try:
    importlib.import_module("mcp")
except ImportError as e:
    verbose_logger.debug(f"MCP module not found: {e}")
    MCP_AVAILABLE = False

if MCP_AVAILABLE:
    from litellm.proxy._experimental.mcp_server.db import (
        create_mcp_server,
        delete_mcp_server,
        get_all_mcp_servers,
        get_all_mcp_servers_for_user,
        get_mcp_server,
        get_mcp_servers,
        update_mcp_server,
    )
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )
    from litellm.proxy._types import (
        LiteLLM_MCPServerTable,
        LitellmUserRoles,
        NewMCPServerRequest,
        SpecialMCPServerName,
        UpdateMCPServerRequest,
        UserAPIKeyAuth,
    )
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.management_endpoints.common_utils import _user_has_admin_view
    from litellm.proxy.management_helpers.utils import management_endpoint_wrapper

    def get_prisma_client_or_throw(message: str):
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": message},
            )
        return prisma_client

    def does_mcp_server_exist(
        mcp_server_records: Iterable[LiteLLM_MCPServerTable], mcp_server_id: str
    ) -> bool:
        """
        Check if the mcp server with the given id exists in the iterable of mcp servers
        """
        for mcp_server_record in mcp_server_records:
            if mcp_server_record.server_id == mcp_server_id:
                return True
        return False


    # Router to fetch all MCP tools available for the current key

    @router.get(
        "/tools",
        tags=["mcp"],
        dependencies=[Depends(user_api_key_auth)],
    )
    async def get_mcp_tools(
            user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """
        Get all MCP tools available for the current key, including those from access groups
        """
        from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
            MCPRequestHandler,
        )
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        # This now includes both direct and access group servers
        server_ids = await MCPRequestHandler._get_allowed_mcp_servers_for_key(user_api_key_dict)

        tools = []
        for server_id in server_ids:
            tools.extend(await global_mcp_server_manager.get_tools_for_server(server_id))

        verbose_proxy_logger.debug(f"Available tools: {tools}")

        return {"tools": tools}

    @router.get(
        "/access_groups",
        tags=["mcp"],
        dependencies=[Depends(user_api_key_auth)],
    )
    async def get_mcp_access_groups(
            user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """
        Get all available MCP access groups from the database AND config
        """
        from litellm.proxy.proxy_server import prisma_client
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import global_mcp_server_manager

        access_groups = set()

        # Get from config-loaded servers
        for server in global_mcp_server_manager.config_mcp_servers.values():
            if server.access_groups:
                access_groups.update(server.access_groups)

        # Get from DB
        if prisma_client is not None:
            try:
                mcp_servers = await prisma_client.db.litellm_mcpservertable.find_many()
                for server in mcp_servers:
                    if hasattr(server, 'mcp_access_groups') and server.mcp_access_groups:
                        access_groups.update(server.mcp_access_groups)
            except Exception as e:
                verbose_proxy_logger.debug(f"Error getting MCP access groups: {e}")

        # Convert to sorted list
        access_groups_list = sorted(list(access_groups))
        return {"access_groups": access_groups_list}

    ## FastAPI Routes
    @router.get(
        "/server",
        description="Returns the mcp server list with associated teams",
        dependencies=[Depends(user_api_key_auth)],
        response_model=List[LiteLLM_MCPServerTable],
    )
    async def fetch_all_mcp_servers(
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """
        Get all of the configured mcp servers for the user in the db with their associated teams
        ```
        curl --location 'http://localhost:4000/v1/mcp/server' \
        --header 'Authorization: Bearer your_api_key_here'
        ```
        """
        from datetime import datetime

        prisma_client = get_prisma_client_or_throw(
            "Database not connected. Connect a database to your proxy"
        )

        # First, get all relevant teams
        user_teams = []
        teams = await prisma_client.db.litellm_teamtable.find_many(
                include={
                    "object_permission": True
                }
            )
        if _user_has_admin_view(user_api_key_dict):
            # Admin can see all teams
            user_teams = teams
        else:
            for team in teams:
                if team.members_with_roles:
                    for member in team.members_with_roles:
                        if (
                            "user_id" in member
                            and member["user_id"] is not None
                            and member["user_id"] == user_api_key_dict.user_id
                        ):
                            user_teams.append(team)

        # Create a mapping of server_id to teams that have access to it
        server_to_teams_map: Dict[str, List[Dict[str, str]]] = {}
        for team in user_teams:
            if team.object_permission and team.object_permission.mcp_servers:
                for server_id in team.object_permission.mcp_servers:
                    if server_id not in server_to_teams_map:
                        server_to_teams_map[server_id] = []
                    server_to_teams_map[server_id].append({
                        "team_id": team.team_id,
                        "team_alias": team.team_alias,
                        "organization_id": team.organization_id
                    })

        # Get MCP servers

        # get all mcp server_ids from user_teams
        mcp_server_ids = []
        for team in user_teams:
            if team.object_permission and team.object_permission.mcp_servers:
                mcp_server_ids.extend(team.object_permission.mcp_servers)

        LIST_MCP_SERVERS = await get_mcp_servers(prisma_client, mcp_server_ids)

        if _user_has_admin_view(user_api_key_dict):
            all_mcp_servers = await get_all_mcp_servers(prisma_client)
            # add all_mcp_servers to LIST_MCP_SERVERS such that there are no duplicates
            for server in all_mcp_servers:
                if server.server_id not in mcp_server_ids:
                    LIST_MCP_SERVERS.append(server)

        # Add config.yaml servers
        ALLOWED_MCP_SERVER_IDS = (
            await global_mcp_server_manager.get_allowed_mcp_servers(
                user_api_key_auth=user_api_key_dict
            )
        )
        ALL_CONFIG_MCP_SERVERS = global_mcp_server_manager.config_mcp_servers
        for _server_id, _server_config in ALL_CONFIG_MCP_SERVERS.items():
            if _server_id in ALLOWED_MCP_SERVER_IDS:
                LIST_MCP_SERVERS.append(
                    LiteLLM_MCPServerTable(
                        server_id=_server_id,
                        server_name=_server_config.name,
                        alias=_server_config.alias,
                        url=_server_config.url,
                        transport=_server_config.transport,
                        spec_version=_server_config.spec_version,
                        auth_type=_server_config.auth_type,
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                        mcp_info=_server_config.mcp_info,
                        # Stdio-specific fields
                        command=getattr(_server_config, 'command', None),
                        args=getattr(_server_config, 'args', None) or [],
                        env=getattr(_server_config, 'env', None) or {},
                    )
                )

        # Map servers to their teams and return
        LIST_MCP_SERVERS = [
            LiteLLM_MCPServerTable(
                server_id=server.server_id,
                server_name=server.server_name,
                alias=server.alias,
                description=server.description,
                url=server.url,
                transport=server.transport,
                spec_version=server.spec_version,
                auth_type=server.auth_type,
                created_at=server.created_at,
                created_by=server.created_by,
                updated_at=server.updated_at,
                updated_by=server.updated_by,
                mcp_access_groups=server.mcp_access_groups if server.mcp_access_groups is not None else [],
                mcp_info=server.mcp_info,
                teams=cast(List[Dict[str, str | None]], server_to_teams_map.get(server.server_id, [])),
                # Stdio-specific fields
                command=getattr(server, 'command', None),
                args=getattr(server, 'args', None) or [],
                env=getattr(server, 'env', None) or {},
            )
            for server in LIST_MCP_SERVERS
        ]

        return LIST_MCP_SERVERS

    @router.get(
        "/server/{server_id}",
        description="Returns the mcp server info",
        dependencies=[Depends(user_api_key_auth)],
        response_model=LiteLLM_MCPServerTable,
    )
    async def fetch_mcp_server(
        server_id: str,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """
        Get the info on the mcp server specified by the `server_id`
        Parameters:
        - server_id: str - Required. The unique identifier of the mcp server to get info on.
        ```
        curl --location 'http://localhost:4000/v1/mcp/server/server_id' \
        --header 'Authorization: Bearer your_api_key_here'
        ```
        """
        prisma_client = get_prisma_client_or_throw(
            "Database not connected. Connect a database to your proxy"
        )

        # check to see if server exists for all users
        mcp_server = await get_mcp_server(prisma_client, server_id)
        if mcp_server is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": f"MCP Server with id {server_id} not found"},
            )

        # Implement authz restriction from requested user
        if _user_has_admin_view(user_api_key_dict):
            return mcp_server

        # Perform authz check to filter the mcp servers user has access to
        mcp_server_records = await get_all_mcp_servers_for_user(
            prisma_client, user_api_key_dict
        )
        exists = does_mcp_server_exist(mcp_server_records, server_id)

        if exists:
            global_mcp_server_manager.add_update_server(mcp_server)
            return mcp_server
        else:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": f"User does not have permission to view mcp server with id {server_id}. You can only view mcp servers that you have access to."
                },
            )

    @router.post(
        "/server",
        description="Allows creation of mcp servers",
        dependencies=[Depends(user_api_key_auth)],
        response_model=LiteLLM_MCPServerTable,
        status_code=status.HTTP_201_CREATED,
    )
    @management_endpoint_wrapper
    async def add_mcp_server(
        payload: NewMCPServerRequest,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
        litellm_changed_by: Optional[str] = Header(
            None,
            description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
        ),
    ):
        """
        Allow users to add a new external mcp server.
        """
        prisma_client = get_prisma_client_or_throw(
            "Database not connected. Connect a database to your proxy"
        )

        # Validate and normalize payload fields
        validate_and_normalize_mcp_server_payload(payload)

        # AuthZ - restrict only proxy admins to create mcp servers
        if LitellmUserRoles.PROXY_ADMIN != user_api_key_dict.user_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "User does not have permission to create mcp servers. You can only create mcp servers if you are a PROXY_ADMIN."
                },
            )
        elif payload.server_id is not None:
            # fail if the mcp server with id already exists
            mcp_server = await get_mcp_server(prisma_client, payload.server_id)
            if mcp_server is not None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": f"MCP Server with id {payload.server_id} already exists. Cannot create another."
                    },
                )
        elif (
            SpecialMCPServerName.all_team_servers == payload.server_id
            or SpecialMCPServerName.all_proxy_servers == payload.server_id
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": f"MCP Server with id {payload.server_id} is special and cannot be used."
                },
            )

        # TODO: audit log for create

        # Attempt to create the mcp server
        try:
            new_mcp_server = await create_mcp_server(
                prisma_client,
                payload,
                touched_by=user_api_key_dict.user_id or LITELLM_PROXY_ADMIN_NAME,
            )
            global_mcp_server_manager.add_update_server(new_mcp_server)
        except Exception as e:
            verbose_proxy_logger.exception(f"Error creating mcp server: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": f"Error creating mcp server: {str(e)}"},
            )
        return new_mcp_server

    @router.delete(
        "/server/{server_id}",
        description="Allows deleting mcp serves in the db",
        dependencies=[Depends(user_api_key_auth)],
        response_class=JSONResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    @management_endpoint_wrapper
    async def remove_mcp_server(
        server_id: str,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
        litellm_changed_by: Optional[str] = Header(
            None,
            description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
        ),
    ):
        """
        Delete MCP Server from db and associated MCP related server entities.

        Parameters:
        - server_id: str - Required. The unique identifier of the mcp server to delete.
        ```
        curl -X "DELETE" --location 'http://localhost:4000/v1/mcp/server/server_id' \
        --header 'Authorization: Bearer your_api_key_here'
        ```
        """
        prisma_client = get_prisma_client_or_throw(
            "Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
        )

        # Authz - restrict only admins to delete mcp servers
        if LitellmUserRoles.PROXY_ADMIN != user_api_key_dict.user_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "Call not allowed to delete MCP server. User is not a proxy admin. route={}".format(
                        "DELETE /v1/mcp/server"
                    )
                },
            )

        # try to delete the mcp server
        mcp_server_record_deleted = await delete_mcp_server(prisma_client, server_id)

        if mcp_server_record_deleted is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": f"MCP Server not found, passed server_id={server_id}"},
            )
        global_mcp_server_manager.remove_server(mcp_server_record_deleted)

        # TODO: Enterprise: Finish audit log trail
        if litellm.store_audit_logs:
            pass

        # TODO: Delete from virtual keys

        # TODO: Delete from teams

        # Update from global mcp store

        return Response(status_code=status.HTTP_202_ACCEPTED)

    @router.put(
        "/server",
        description="Allows deleting mcp serves in the db",
        dependencies=[Depends(user_api_key_auth)],
        response_model=LiteLLM_MCPServerTable,
        status_code=status.HTTP_202_ACCEPTED,
    )
    @management_endpoint_wrapper
    async def edit_mcp_server(
        payload: UpdateMCPServerRequest,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
        litellm_changed_by: Optional[str] = Header(
            None,
            description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
        ),
    ):
        """
        Updates the MCP Server in the db.

        Parameters:
        - payload: UpdateMCPServerRequest - Required. The updated mcp server data.
        ```
        curl -X "PUT" --location 'http://localhost:4000/v1/mcp/server' \
        --header 'Authorization: Bearer your_api_key_here'
        ```
        """
        prisma_client = get_prisma_client_or_throw(
            "Database not connected. Connect a database to your proxy - https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
        )

        # Validate and normalize payload fields
        validate_and_normalize_mcp_server_payload(payload)

        # Authz - restrict only admins to delete mcp servers
        if LitellmUserRoles.PROXY_ADMIN != user_api_key_dict.user_role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "Call not allowed to update MCP server. User is not a proxy admin. route={}".format(
                        "PUT /v1/mcp/server"
                    )
                },
            )

        # try to update the mcp server
        mcp_server_record_updated = await update_mcp_server(
            prisma_client,
            payload,
            touched_by=user_api_key_dict.user_id or LITELLM_PROXY_ADMIN_NAME,
        )

        if mcp_server_record_updated is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": f"MCP Server not found, passed server_id={payload.server_id}"
                },
            )
        global_mcp_server_manager.add_update_server(mcp_server_record_updated)

        # TODO: Enterprise: Finish audit log trail
        if litellm.store_audit_logs:
            pass

        return mcp_server_record_updated

