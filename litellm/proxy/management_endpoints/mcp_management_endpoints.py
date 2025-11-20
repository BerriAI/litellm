"""
1. Allow proxy admin to perform create, update, and delete operations on MCP servers in the db.
2. Allows users to view the mcp servers they have access to.

Endpoints here:
- GET `/v1/mcp/server` - Returns all of the configured mcp servers in the db filtered by requestor's access
- GET `/v1/mcp/server/{server_id}` - Returns the the specific mcp server in the db given `server_id` filtered by requestor's access
- POST `/v1/mcp/server` - Add a new external mcp server.
- PUT `/v1/mcp/server` -  Edits an existing mcp server.
- DELETE `/v1/mcp/server/{server_id}` - Deletes the mcp server given `server_id`.
- GET `/v1/mcp/tools - lists all the tools available for a key
- GET `/v1/mcp/access_groups` - lists all available MCP access groups

"""

import importlib
from datetime import datetime
from typing import Iterable, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Response, status
from fastapi.responses import JSONResponse

import litellm
from litellm._logging import verbose_logger, verbose_proxy_logger
from litellm.constants import LITELLM_PROXY_ADMIN_NAME
from litellm.proxy._experimental.mcp_server.utils import (
    validate_and_normalize_mcp_server_payload,
)

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
        get_all_mcp_servers_for_user,
        get_mcp_server,
        update_mcp_server,
    )
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )
    from litellm.proxy._types import (
        LiteLLM_MCPServerTable,
        LitellmUserRoles,
        MakeMCPServersPublicRequest,
        NewMCPServerRequest,
        SpecialMCPServerName,
        UpdateMCPServerRequest,
        UserAPIKeyAuth,
    )
    from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
    from litellm.proxy.management_endpoints.common_utils import _user_has_admin_view
    from litellm.proxy.management_helpers.utils import management_endpoint_wrapper
    from litellm.types.mcp_server.mcp_server_manager import MCPInfo

    def _redact_mcp_credentials(
        mcp_server: LiteLLM_MCPServerTable,
    ) -> LiteLLM_MCPServerTable:
        """Return a copy of the MCP server object with credentials removed."""

        try:
            redacted_server = mcp_server.model_copy(deep=True)
        except AttributeError:
            redacted_server = mcp_server.copy(deep=True)  # type: ignore[attr-defined]

        if hasattr(redacted_server, "credentials"):
            setattr(redacted_server, "credentials", None)

        return redacted_server

    def _redact_mcp_credentials_list(
        mcp_servers: Iterable[LiteLLM_MCPServerTable],
    ) -> List[LiteLLM_MCPServerTable]:
        return [_redact_mcp_credentials(server) for server in mcp_servers]

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
        from litellm.proxy._experimental.mcp_server.server import _list_mcp_tools

        tools = await _list_mcp_tools(
            user_api_key_auth=user_api_key_dict,
            mcp_auth_header=None,
            mcp_servers=None,
            mcp_server_auth_headers=None,
        )
        dumped_tools = [dict(tool) for tool in tools]

        return {"tools": dumped_tools}

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
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.proxy.proxy_server import prisma_client

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
                    if (
                        hasattr(server, "mcp_access_groups")
                        and server.mcp_access_groups
                    ):
                        access_groups.update(server.mcp_access_groups)
            except Exception as e:
                verbose_proxy_logger.debug(f"Error getting MCP access groups: {e}")

        # Convert to sorted list
        access_groups_list = sorted(list(access_groups))
        return {"access_groups": access_groups_list}

    @router.get(
        "/server/{server_id}/health",
        description="Perform health check on a specific MCP server",
        dependencies=[Depends(user_api_key_auth)],
    )
    async def health_check_mcp_server(
        server_id: str,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """
        Perform a health check on the MCP server specified by the `server_id`
        Parameters:
        - server_id: str - Required. The unique identifier of the mcp server to health check.
        ```
        curl --location 'http://localhost:4000/v1/mcp/server/{server_id}/health' \
        --header 'Authorization: Bearer your_api_key_here'
        ```
        """
        # Check if server exists and user has access
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
        if not _user_has_admin_view(user_api_key_dict):
            # Perform authz check to filter the mcp servers user has access to
            mcp_server_records = await get_all_mcp_servers_for_user(
                prisma_client, user_api_key_dict
            )
            exists = does_mcp_server_exist(mcp_server_records, server_id)

            if not exists:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "error": f"User does not have permission to access mcp server with id {server_id}. You can only access mcp servers that you have access to."
                    },
                )

        # Perform health check using server manager
        try:
            health_result = await global_mcp_server_manager.health_check_server(
                server_id
            )
            return health_result
        except Exception as e:
            verbose_proxy_logger.exception(
                f"Error performing health check on MCP server {server_id}: {str(e)}"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": f"Error performing health check: {str(e)}"},
            )

    @router.get(
        "/server/health",
        description="Perform health check on all accessible MCP servers",
        dependencies=[Depends(user_api_key_auth)],
    )
    async def health_check_all_mcp_servers(
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """
        Perform health checks on all MCP servers accessible to the user
        ```
        curl --location 'http://localhost:4000/v1/mcp/server/health' \
        --header 'Authorization: Bearer your_api_key_here'
        ```
        """
        # Use server manager to get health checks for allowed servers
        try:
            all_health_results = (
                await global_mcp_server_manager.health_check_allowed_servers(
                    user_api_key_auth=user_api_key_dict
                )
            )

            return {
                "total_servers": len(all_health_results),
                "healthy_count": len(
                    [r for r in all_health_results.values() if r["status"] == "healthy"]
                ),
                "unhealthy_count": len(
                    [
                        r
                        for r in all_health_results.values()
                        if r["status"] == "unhealthy"
                    ]
                ),
                "unknown_count": len(
                    [r for r in all_health_results.values() if r["status"] == "unknown"]
                ),
                "servers": all_health_results,
            }
        except Exception as e:
            verbose_proxy_logger.exception(
                f"Error performing health checks on MCP servers: {str(e)}"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": f"Error performing health checks: {str(e)}"},
            )

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

        # Use server manager to get all servers with health and team data
        mcp_servers = (
            await global_mcp_server_manager.get_all_mcp_servers_with_health_and_teams(
                user_api_key_auth=user_api_key_dict
            )
        )
        redacted_mcp_servers = _redact_mcp_credentials_list(mcp_servers)

        # augment the mcp servers with public status
        if litellm.public_mcp_servers is not None:
            for server in redacted_mcp_servers:
                if server.server_id in litellm.public_mcp_servers:
                    if server.mcp_info is None:
                        server.mcp_info = {}
                    server.mcp_info["is_public"] = True
        return redacted_mcp_servers

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

        # Perform health check on the server using server manager
        try:
            health_result = await global_mcp_server_manager.health_check_server(
                server_id
            )
            # Update the server object with health check results
            mcp_server.status = health_result.get("status", "unknown")
            mcp_server.last_health_check = (
                datetime.fromisoformat(
                    health_result.get("last_health_check", datetime.now().isoformat())
                )
                if health_result.get("last_health_check")
                else None
            )
            mcp_server.health_check_error = health_result.get("error")
        except Exception as e:
            verbose_proxy_logger.debug(
                f"Error performing health check on server {server_id}: {e}"
            )
            mcp_server.status = "unknown"
            mcp_server.last_health_check = datetime.now()
            mcp_server.health_check_error = str(e)

        # Implement authz restriction from requested user
        if _user_has_admin_view(user_api_key_dict):
            return _redact_mcp_credentials(mcp_server)

        # Perform authz check to filter the mcp servers user has access to
        mcp_server_records = await get_all_mcp_servers_for_user(
            prisma_client, user_api_key_dict
        )
        exists = does_mcp_server_exist(mcp_server_records, server_id)

        if exists:
            global_mcp_server_manager.add_update_server(mcp_server)
            return _redact_mcp_credentials(mcp_server)
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

            # Ensure registry is up to date by reloading from database
            await global_mcp_server_manager.reload_servers_from_database()
        except Exception as e:
            verbose_proxy_logger.exception(f"Error creating mcp server: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": f"Error creating mcp server: {str(e)}"},
            )
        return _redact_mcp_credentials(new_mcp_server)

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

        # Ensure registry is up to date by reloading from database
        await global_mcp_server_manager.reload_servers_from_database()

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

        # Ensure registry is up to date by reloading from database
        await global_mcp_server_manager.reload_servers_from_database()

        # TODO: Enterprise: Finish audit log trail
        if litellm.store_audit_logs:
            pass

        return _redact_mcp_credentials(mcp_server_record_updated)

    @router.post(
        "/make_public",
        description="Allows making MCP servers public for AI Hub",
        dependencies=[Depends(user_api_key_auth)],
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def make_mcp_servers_public(
        request: MakeMCPServersPublicRequest,
        user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    ):
        """
        Make MCP servers public for AI Hub
        """
        try:
            # Update the public model groups
            import litellm
            from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
                global_mcp_server_manager,
            )
            from litellm.proxy.proxy_server import proxy_config

            # Load existing config
            config = await proxy_config.get_config()
            # Check if user has admin permissions
            if user_api_key_dict.user_role != LitellmUserRoles.PROXY_ADMIN:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "Only proxy admins can update public mcp servers. Your role={}".format(
                            user_api_key_dict.user_role
                        )
                    },
                )

            if litellm.public_mcp_servers is None:
                litellm.public_mcp_servers = []

            for server_id in request.mcp_server_ids:
                server = global_mcp_server_manager.get_mcp_server_by_id(
                    server_id=server_id
                )
                if server is None:
                    raise HTTPException(
                        status_code=404,
                        detail=f"MCP Server with ID {server_id} not found",
                    )

            litellm.public_mcp_servers = request.mcp_server_ids

            # Update config with new settings
            if "litellm_settings" not in config or config["litellm_settings"] is None:
                config["litellm_settings"] = {}

            config["litellm_settings"][
                "public_mcp_servers"
            ] = litellm.public_mcp_servers

            # Save the updated config
            await proxy_config.save_config(new_config=config)

            verbose_proxy_logger.debug(
                f"Updated public mcp servers to: {litellm.public_mcp_servers} by user: {user_api_key_dict.user_id}"
            )

            return {
                "message": "Successfully updated public mcp servers",
                "public_mcp_servers": litellm.public_mcp_servers,
                "updated_by": user_api_key_dict.user_id,
            }
        except HTTPException:
            raise
        except Exception as e:
            verbose_proxy_logger.exception(f"Error making agent public: {e}")
            raise HTTPException(status_code=500, detail=str(e))
