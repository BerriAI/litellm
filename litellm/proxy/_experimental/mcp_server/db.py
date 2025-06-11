import uuid
from typing import Iterable, List, Optional, Set

from litellm.proxy._types import (
    LiteLLM_MCPServerTable,
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamTable,
    NewMCPServerRequest,
    SpecialMCPServerName,
    UpdateMCPServerRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.utils import PrismaClient


async def get_all_mcp_servers(
    prisma_client: PrismaClient,
) -> List[LiteLLM_MCPServerTable]:
    """
    Returns all of the mcp servers from the db
    """
    mcp_servers = await prisma_client.db.litellm_mcpservertable.find_many()

    return mcp_servers


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
    mcp_servers: List[
        LiteLLM_MCPServerTable
    ] = await prisma_client.db.litellm_mcpservertable.find_many(
        where={
            "server_id": {"in": server_ids},
        }
    )
    return mcp_servers


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

    mcp_server_record = await prisma_client.db.litellm_mcpservertable.create(
        data={
            **data.model_dump(),
            "created_by": touched_by,
            "updated_by": touched_by,
        }
    )
    return mcp_server_record


async def update_mcp_server(
    prisma_client: PrismaClient, data: UpdateMCPServerRequest, touched_by: str
) -> LiteLLM_MCPServerTable:
    """
    Update a new mcp server record in the db
    """
    mcp_server_record = await prisma_client.db.litellm_mcpservertable.update(
        where={
            "server_id": data.server_id,
        },
        data={
            **data.model_dump(),
            "created_by": touched_by,
            "updated_by": touched_by,
        },
    )
    return mcp_server_record
