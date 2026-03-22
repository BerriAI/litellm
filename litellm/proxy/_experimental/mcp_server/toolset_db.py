import json
from typing import List, Optional

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.proxy.utils import PrismaClient
from litellm.types.mcp_server.mcp_toolset import (
    MCPToolset,
    NewMCPToolsetRequest,
    UpdateMCPToolsetRequest,
)


def _toolset_from_row(row) -> MCPToolset:
    data = row.model_dump()
    tools = data.get("tools") or []
    if isinstance(tools, str):
        tools = json.loads(tools)
    data["tools"] = tools
    return MCPToolset(**data)


async def create_mcp_toolset(
    prisma_client: PrismaClient,
    data: NewMCPToolsetRequest,
    touched_by: str,
) -> MCPToolset:
    data_dict = data.model_dump(exclude_none=True)
    data_dict["toolset_id"] = str(uuid.uuid4())
    data_dict["tools"] = json.dumps(data_dict.get("tools", []))
    data_dict["created_by"] = touched_by
    data_dict["updated_by"] = touched_by
    row = await prisma_client.db.litellm_mcptoolsettable.create(data=data_dict)
    return _toolset_from_row(row)


async def get_mcp_toolset(
    prisma_client: PrismaClient,
    toolset_id: str,
) -> Optional[MCPToolset]:
    row = await prisma_client.db.litellm_mcptoolsettable.find_unique(
        where={"toolset_id": toolset_id}
    )
    if row is None:
        return None
    return _toolset_from_row(row)


async def list_mcp_toolsets(
    prisma_client: PrismaClient,
    toolset_ids: Optional[List[str]] = None,
) -> List[MCPToolset]:
    try:
        where = {}
        if toolset_ids is not None:
            where = {"toolset_id": {"in": toolset_ids}}
        rows = await prisma_client.db.litellm_mcptoolsettable.find_many(where=where)
        return [_toolset_from_row(r) for r in rows]
    except Exception as e:
        verbose_proxy_logger.debug(
            "litellm.proxy._experimental.mcp_server.toolset_db::list_mcp_toolsets - {}".format(
                str(e)
            )
        )
        return []


async def get_mcp_toolset_by_name(
    prisma_client: PrismaClient,
    toolset_name: str,
) -> Optional[MCPToolset]:
    row = await prisma_client.db.litellm_mcptoolsettable.find_first(
        where={"toolset_name": toolset_name}
    )
    if row is None:
        return None
    return _toolset_from_row(row)


async def update_mcp_toolset(
    prisma_client: PrismaClient,
    data: UpdateMCPToolsetRequest,
    touched_by: str,
) -> MCPToolset:
    data_dict = data.model_dump(exclude_none=True, exclude={"toolset_id"})
    if "tools" in data_dict:
        data_dict["tools"] = json.dumps(data_dict["tools"])
    data_dict["updated_by"] = touched_by
    row = await prisma_client.db.litellm_mcptoolsettable.update(
        where={"toolset_id": data.toolset_id},
        data=data_dict,
    )
    return _toolset_from_row(row)


async def delete_mcp_toolset(
    prisma_client: PrismaClient,
    toolset_id: str,
) -> Optional[MCPToolset]:
    row = await prisma_client.db.litellm_mcptoolsettable.delete(
        where={"toolset_id": toolset_id}
    )
    if row is None:
        return None
    return _toolset_from_row(row)
