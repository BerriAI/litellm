import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Final, Protocol

from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.proxy.utils import PrismaClient
from litellm.repositories.table_repositories import MCPToolsetRepository
from litellm.types.mcp_server.mcp_toolset import (
    MCPToolset,
    MCPToolsetTool,
    NewMCPToolsetRequest,
    UpdateMCPToolsetRequest,
)


class MCPToolsetFields(TypedDict):
    """The ``MCPToolset`` constructor keywords a toolset row expands into."""

    toolset_id: ReadOnly[str]
    toolset_name: ReadOnly[str]
    description: NotRequired[ReadOnly[str | None]]
    tools: NotRequired[ReadOnly[list[MCPToolsetTool]]]
    created_at: NotRequired[ReadOnly[datetime | None]]
    created_by: NotRequired[ReadOnly[str | None]]
    updated_at: NotRequired[ReadOnly[datetime | None]]
    updated_by: NotRequired[ReadOnly[str | None]]


class MCPToolsetRowData(TypedDict):
    """A toolset table row, whose ``tools`` column is stored as JSON."""

    toolset_id: ReadOnly[str]
    toolset_name: ReadOnly[str]
    description: NotRequired[ReadOnly[str | None]]
    tools: NotRequired[ReadOnly[str | list[MCPToolsetTool]]]
    created_at: NotRequired[ReadOnly[datetime | None]]
    created_by: NotRequired[ReadOnly[str | None]]
    updated_at: NotRequired[ReadOnly[datetime | None]]
    updated_by: NotRequired[ReadOnly[str | None]]


class MCPToolsetRow(Protocol):
    """A row of the toolset table, as the prisma client returns it."""

    def model_dump(self) -> MCPToolsetRowData: ...


class MCPToolsetTable(Protocol):
    """The prisma table actions this module runs against the toolset table."""

    async def create(self, data: Mapping[str, object]) -> MCPToolsetRow: ...

    async def find_unique(self, where: Mapping[str, object]) -> MCPToolsetRow | None: ...

    async def find_first(self, where: Mapping[str, object]) -> MCPToolsetRow | None: ...

    async def find_many(self, where: Mapping[str, object]) -> Sequence[MCPToolsetRow]: ...

    async def update(self, where: Mapping[str, object], data: Mapping[str, object]) -> MCPToolsetRow: ...

    async def delete(self, where: Mapping[str, object]) -> MCPToolsetRow: ...


def _toolset_table(prisma_client: PrismaClient) -> MCPToolsetTable:
    """The toolset table actions of the prisma client."""
    return MCPToolsetRepository(prisma_client).table


def _toolset_from_row(row: MCPToolsetRow) -> MCPToolset:
    data: Final = row.model_dump()
    tools: Final = data.get("tools") or []
    resolved: Final[MCPToolsetFields] = {
        **data,
        "tools": json.loads(tools) if isinstance(tools, str) else tools,
    }
    return MCPToolset(**resolved)


async def create_mcp_toolset(
    prisma_client: PrismaClient,
    data: NewMCPToolsetRequest,
    touched_by: str,
) -> MCPToolset:
    data_dict: Final = data.model_dump(exclude_none=True)
    data_dict["toolset_id"] = str(uuid.uuid4())
    data_dict["tools"] = json.dumps(data_dict.get("tools", []))
    data_dict["created_by"] = touched_by
    data_dict["updated_by"] = touched_by
    row: Final = await _toolset_table(prisma_client).create(data=data_dict)
    return _toolset_from_row(row)


async def get_mcp_toolset(
    prisma_client: PrismaClient,
    toolset_id: str,
) -> MCPToolset | None:
    row: Final = await _toolset_table(prisma_client).find_unique(where={"toolset_id": toolset_id})
    if row is None:
        return None
    return _toolset_from_row(row)


async def list_mcp_toolsets(
    prisma_client: PrismaClient,
    toolset_ids: Sequence[str] | None = None,
) -> Sequence[MCPToolset]:
    try:
        where: Final[Mapping[str, object]] = {} if toolset_ids is None else {"toolset_id": {"in": toolset_ids}}
        rows: Final = await _toolset_table(prisma_client).find_many(where=where)
        return [_toolset_from_row(r) for r in rows]
    except Exception as e:
        verbose_proxy_logger.warning("litellm.proxy._experimental.mcp_server.toolset_db::list_mcp_toolsets - %s", e)
        return []


async def get_mcp_toolset_by_name(
    prisma_client: PrismaClient,
    toolset_name: str,
) -> MCPToolset | None:
    row: Final = await _toolset_table(prisma_client).find_first(where={"toolset_name": toolset_name})
    if row is None:
        return None
    return _toolset_from_row(row)


async def update_mcp_toolset(
    prisma_client: PrismaClient,
    data: UpdateMCPToolsetRequest,
    touched_by: str,
) -> MCPToolset | None:
    data_dict: Final = data.model_dump(exclude_none=True, exclude={"toolset_id"})
    if "tools" in data_dict:
        data_dict["tools"] = json.dumps(data_dict["tools"])
    data_dict["updated_by"] = touched_by
    try:
        row: Final = await _toolset_table(prisma_client).update(
            where={"toolset_id": data.toolset_id},
            data=data_dict,
        )
    except Exception as e:
        from prisma.errors import RecordNotFoundError

        if isinstance(e, RecordNotFoundError):
            return None
        raise
    return _toolset_from_row(row)


async def delete_mcp_toolset(
    prisma_client: PrismaClient,
    toolset_id: str,
) -> MCPToolset | None:
    try:
        row: Final = await _toolset_table(prisma_client).delete(where={"toolset_id": toolset_id})
    except Exception as e:
        from prisma.errors import RecordNotFoundError

        if isinstance(e, RecordNotFoundError):
            return None
        raise
    return _toolset_from_row(row)
