from typing import Any, Dict, Iterable, List, Optional, Set, Union, cast

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.proxy._types import (
    LiteLLM_MCPServerTable,
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamTable,
    NewMCPServerRequest,
    SpecialMCPServerName,
    UpdateMCPServerRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    _get_salt_key,
    encrypt_value_helper,
)
from litellm.proxy.utils import PrismaClient
from litellm.types.mcp import MCPCredentials


def _prepare_mcp_server_data(
    data: Union[NewMCPServerRequest, UpdateMCPServerRequest],
) -> Dict[str, Any]:
    """
    Helper function to prepare MCP server data for database operations.
    Handles JSON field serialization for mcp_info and env fields.

    Args:
        data: NewMCPServerRequest or UpdateMCPServerRequest object

    Returns:
        Dict with properly serialized JSON fields
    """
    from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

    # Convert model to dict
    data_dict = data.model_dump(exclude_none=True)
    # Ensure alias is always present in the dict (even if None)
    if "alias" not in data_dict:
        data_dict["alias"] = getattr(data, "alias", None)

    # Handle credentials serialization
    credentials = data_dict.get("credentials")
    if credentials is not None:
        data_dict["credentials"] = encrypt_credentials(
            credentials=credentials, encryption_key=_get_salt_key()
        )
        data_dict["credentials"] = safe_dumps(data_dict["credentials"])

    # Handle static_headers serialization
    if data.static_headers is not None:
        data_dict["static_headers"] = safe_dumps(data.static_headers)

    # Handle mcp_info serialization
    if data.mcp_info is not None:
        data_dict["mcp_info"] = safe_dumps(data.mcp_info)

    # Handle env serialization
    if data.env is not None:
        data_dict["env"] = safe_dumps(data.env)

    # mcp_access_groups is already List[str], no serialization needed

    return data_dict


def encrypt_credentials(
    credentials: MCPCredentials, encryption_key: Optional[str]
) -> MCPCredentials:
    auth_value = credentials.get("auth_value")
    if auth_value is not None:
        credentials["auth_value"] = encrypt_value_helper(
            value=auth_value,
            new_encryption_key=encryption_key,
        )
    client_id = credentials.get("client_id")
    if client_id is not None:
        credentials["client_id"] = encrypt_value_helper(
            value=client_id,
            new_encryption_key=encryption_key,
        )
    client_secret = credentials.get("client_secret")
    if client_secret is not None:
        credentials["client_secret"] = encrypt_value_helper(
            value=client_secret,
            new_encryption_key=encryption_key,
        )
    return credentials


async def get_all_mcp_servers(
    prisma_client: PrismaClient,
) -> List[LiteLLM_MCPServerTable]:
    """
    Returns all of the mcp servers from the db
    """
    try:
        mcp_servers = await prisma_client.db.litellm_mcpservertable.find_many()

        return [
            LiteLLM_MCPServerTable(**mcp_server.model_dump())
            for mcp_server in mcp_servers
        ]
    except Exception as e:
        verbose_proxy_logger.debug(
            "litellm.proxy._experimental.mcp_server.db.py::get_all_mcp_servers - {}".format(
                str(e)
            )
        )
        return []


async def get_mcp_server(
    prisma_client: PrismaClient, server_id: str
) -> Optional[LiteLLM_MCPServerTable]:
    """
    Returns the matching mcp server from the db iff exists
    """
    mcp_server: Optional[
        LiteLLM_MCPServerTable
    ] = await prisma_client.db.litellm_mcpservertable.find_unique(
        where={
            "server_id": server_id,
        }
    )
    return mcp_server


async def get_mcp_servers(
    prisma_client: PrismaClient, server_ids: Iterable[str]
) -> List[LiteLLM_MCPServerTable]:
    """
    Returns the matching mcp servers from the db with the server_ids
    """
    _mcp_servers: List[
        LiteLLM_MCPServerTable
    ] = await prisma_client.db.litellm_mcpservertable.find_many(
        where={
            "server_id": {"in": server_ids},
        }
    )
    final_mcp_servers: List[LiteLLM_MCPServerTable] = []
    for _mcp_server in _mcp_servers:
        final_mcp_servers.append(LiteLLM_MCPServerTable(**_mcp_server.model_dump()))

    return final_mcp_servers


async def get_mcp_servers_by_verificationtoken(
    prisma_client: PrismaClient, token: str
) -> List[str]:
    """
    Returns the mcp servers from the db for the verification token
    """
    verification_token_record: LiteLLM_TeamTable = (
        await prisma_client.db.litellm_verificationtoken.find_unique(
            where={
                "token": token,
            },
            include={
                "object_permission": True,
            },
        )
    )

    mcp_servers: Optional[List[str]] = []
    if (
        verification_token_record is not None
        and verification_token_record.object_permission is not None
    ):
        mcp_servers = verification_token_record.object_permission.mcp_servers
    return mcp_servers or []


async def get_mcp_servers_by_team(
    prisma_client: PrismaClient, team_id: str
) -> List[str]:
    """
    Returns the mcp servers from the db for the team id
    """
    team_record: LiteLLM_TeamTable = (
        await prisma_client.db.litellm_teamtable.find_unique(
            where={
                "team_id": team_id,
            },
            include={
                "object_permission": True,
            },
        )
    )

    mcp_servers: Optional[List[str]] = []
    if team_record is not None and team_record.object_permission is not None:
        mcp_servers = team_record.object_permission.mcp_servers
    return mcp_servers or []


async def get_all_mcp_servers_for_user(
    prisma_client: PrismaClient,
    user: UserAPIKeyAuth,
) -> List[LiteLLM_MCPServerTable]:
    """
    Get all the mcp servers filtered by the given user has access to.

    Following Least-Privilege Principle - the requestor should only be able to see the mcp servers that they have access to.
    """

    mcp_server_ids: Set[str] = set()
    mcp_servers = []

    # Get the mcp servers for the key
    if user.api_key:
        token_mcp_servers = await get_mcp_servers_by_verificationtoken(
            prisma_client, user.api_key
        )
        mcp_server_ids.update(token_mcp_servers)

        # check for special team membership
        if (
            SpecialMCPServerName.all_team_servers in mcp_server_ids
            and user.team_id is not None
        ):
            team_mcp_servers = await get_mcp_servers_by_team(
                prisma_client, user.team_id
            )
            mcp_server_ids.update(team_mcp_servers)

    if len(mcp_server_ids) > 0:
        mcp_servers = await get_mcp_servers(prisma_client, mcp_server_ids)

    return mcp_servers


async def get_objectpermissions_for_mcp_server(
    prisma_client: PrismaClient, mcp_server_id: str
) -> List[LiteLLM_ObjectPermissionTable]:
    """
    Get all the object permissions records and the associated team and verficiationtoken records that have access to the mcp server
    """
    object_permission_records = (
        await prisma_client.db.litellm_objectpermissiontable.find_many(
            where={
                "mcp_servers": {"has": mcp_server_id},
            },
            include={
                "teams": True,
                "verification_tokens": True,
            },
        )
    )

    return object_permission_records


async def get_virtualkeys_for_mcp_server(
    prisma_client: PrismaClient, server_id: str
) -> List:
    """
    Get all the virtual keys that have access to the mcp server
    """
    virtual_keys = await prisma_client.db.litellm_verificationtoken.find_many(
        where={
            "mcp_servers": {"has": server_id},
        },
    )

    if virtual_keys is None:
        return []
    return virtual_keys


async def delete_mcp_server_from_team(prisma_client: PrismaClient, server_id: str):
    """
    Remove the mcp server from the team
    """
    pass


async def delete_mcp_server_from_virtualkey():
    """
    Remove the mcp server from the virtual key
    """
    pass


async def delete_mcp_server(
    prisma_client: PrismaClient, server_id: str
) -> Optional[LiteLLM_MCPServerTable]:
    """
    Delete the mcp server from the db by server_id

    Returns the deleted mcp server record if it exists, otherwise None
    """
    deleted_server = await prisma_client.db.litellm_mcpservertable.delete(
        where={
            "server_id": server_id,
        },
    )
    return deleted_server


async def create_mcp_server(
    prisma_client: PrismaClient, data: NewMCPServerRequest, touched_by: str
) -> LiteLLM_MCPServerTable:
    """
    Create a new mcp server record in the db
    """
    if data.server_id is None:
        data.server_id = str(uuid.uuid4())

    # Use helper to prepare data with proper JSON serialization
    data_dict = _prepare_mcp_server_data(data)

    # Add audit fields
    data_dict["created_by"] = touched_by
    data_dict["updated_by"] = touched_by

    new_mcp_server = await prisma_client.db.litellm_mcpservertable.create(
        data=data_dict  # type: ignore
    )

    return new_mcp_server


async def update_mcp_server(
    prisma_client: PrismaClient, data: UpdateMCPServerRequest, touched_by: str
) -> LiteLLM_MCPServerTable:
    """
    Update a new mcp server record in the db
    """
    # Use helper to prepare data with proper JSON serialization
    data_dict = _prepare_mcp_server_data(data)

    # Add audit fields
    data_dict["updated_by"] = touched_by

    updated_mcp_server = await prisma_client.db.litellm_mcpservertable.update(
        where={"server_id": data.server_id}, data=data_dict  # type: ignore
    )

    return updated_mcp_server


async def rotate_mcp_server_credentials_master_key(
    prisma_client: PrismaClient, touched_by: str, new_master_key: str
):
    mcp_servers = await prisma_client.db.litellm_mcpservertable.find_many()

    for mcp_server in mcp_servers:
        credentials = mcp_server.credentials
        if not credentials:
            continue

        credentials_copy = dict(credentials)
        encrypted_credentials = encrypt_credentials(
            credentials=cast(MCPCredentials, credentials_copy),
            encryption_key=new_master_key,
        )

        from litellm.litellm_core_utils.safe_json_dumps import safe_dumps

        serialized_credentials = safe_dumps(encrypted_credentials)

        await prisma_client.db.litellm_mcpservertable.update(
            where={"server_id": mcp_server.server_id},
            data={
                "credentials": serialized_credentials,
                "updated_by": touched_by,
            },
        )
