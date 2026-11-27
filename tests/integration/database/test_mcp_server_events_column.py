import os
import uuid
from typing import Final

import pytest

from litellm.proxy._types import NewMCPServerRequest, UpdateMCPServerRequest
from litellm.proxy.mcp_server.db import _prepare_mcp_server_data
from litellm.proxy.mcp_server.mcp_server_manager import _stored_events_config
from litellm.proxy.utils import PrismaClient
from litellm.types.mcp_server.mcp_server_manager import MCPServerEventsConfig
from tests.integration._support.prisma_clients import connected_proxy_prisma

EVENTS: Final = MCPServerEventsConfig(enabled=True, allowed_events=("build.*",))


async def _stored(client: PrismaClient, server_id: str) -> MCPServerEventsConfig | None:
    row: Final = await client.db.litellm_mcpservertable.find_unique(where={"server_id": server_id})
    return _stored_events_config(row.model_dump().get("events") if row is not None else None)


@pytest.mark.asyncio
async def test_the_events_config_survives_create_partial_update_clear_and_reload() -> None:
    server_id: Final = f"srv-events-{uuid.uuid4().hex}"
    created: Final = _prepare_mcp_server_data(
        NewMCPServerRequest(
            server_id=server_id, server_name=server_id, url="https://upstream.example/mcp", events=EVENTS
        )
    )
    async with connected_proxy_prisma(os.environ["DATABASE_URL"]) as client:
        await client.db.litellm_mcpservertable.create(data={**created, "created_by": "test", "updated_by": "test"})
        after_create: Final = await _stored(client, server_id)

        renamed: Final = UpdateMCPServerRequest(server_id=server_id, description="renamed")
        partial: Final = _prepare_mcp_server_data(renamed, exclude_unset=True, fields_set=renamed.model_fields_set)
        await client.db.litellm_mcpservertable.update(where={"server_id": server_id}, data=partial)
        after_partial: Final = await _stored(client, server_id)

        cleared_request: Final = UpdateMCPServerRequest(server_id=server_id, events=None)
        cleared: Final = _prepare_mcp_server_data(
            cleared_request, exclude_unset=True, fields_set=cleared_request.model_fields_set
        )
        await client.db.litellm_mcpservertable.update(where={"server_id": server_id}, data=cleared)
        after_clear: Final = await _stored(client, server_id)
        await client.db.litellm_mcpservertable.delete(where={"server_id": server_id})
    assert (after_create, after_partial, after_clear) == (EVENTS, EVENTS, None)
