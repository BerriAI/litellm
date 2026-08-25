"""Live e2e: a virtual key's MCP grants survive unrelated /key/update writes.

Two regressions customers hit through the key-edit UI, both driven here exactly as
the UI drives them (POST /key/update with a partial body):

- a budget-only update silently dropped the key's attached MCP servers (fixed in
  PR #34452): /key/update must leave object_permission untouched when the body
  does not carry one
- a key holding ids of since-deleted MCP servers could not be edited at all: the
  update was rejected with "Key requests MCP servers not allowed by team", forcing
  admins to fix DB rows by hand (LIT-3278). Stale ids must be dropped, not fatal
"""

from __future__ import annotations

import pytest

from datadog_mcp import SEARCH_LOGS_TOOL, register_datadog_mcp
from e2e_config import unique_marker
from e2e_http import NoBody, unwrap
from lifecycle import ResourceManager
from mcp_client import McpClient
from models import KeyGenerateBody, KeyInfoParams, KeyInfoResponse, KeyUpdateBody, ObjectPermission

pytestmark = pytest.mark.e2e


def _mcp_key(client: McpClient, resources: ResourceManager, server_ids: list[str]) -> str:
    key = client.proxy.generate_key(
        KeyGenerateBody(
            user_id=f"e2e-mcp-key-perms-{unique_marker()}",
            object_permission=ObjectPermission(mcp_servers=server_ids),
        )
    )
    resources.defer(lambda: client.proxy.delete_key(key))
    return key


def _key_mcp_servers(client: McpClient, key: str) -> list[str]:
    info = unwrap(
        client.proxy.transport.get(
            "/key/info",
            headers=client.proxy.transport.master,
            params=KeyInfoParams(key=key),
            response_type=KeyInfoResponse,
        )
    ).info
    if info.object_permission is None:
        return []
    return info.object_permission.mcp_servers or []


class TestKeyUpdatePreservesMcpGrants:
    @pytest.mark.covers("mcp.key_update.api_key.budget_change_preserves_grants")
    def test_budget_only_update_preserves_mcp_servers(
        self, client: McpClient, resources: ResourceManager
    ) -> None:
        server_id = register_datadog_mcp(client, resources)
        client.await_registered(server_id)
        key = _mcp_key(client, resources, [server_id])
        assert _key_mcp_servers(client, key) == [server_id]
        tool_before = client.await_tool(key, server_id, SEARCH_LOGS_TOOL)

        unwrap(
            client.proxy.transport.post(
                "/key/update",
                headers=client.proxy.transport.master,
                json=KeyUpdateBody(key=key, max_budget=42.0),
                response_type=NoBody,
            )
        )

        info = unwrap(
            client.proxy.transport.get(
                "/key/info",
                headers=client.proxy.transport.master,
                params=KeyInfoParams(key=key),
                response_type=KeyInfoResponse,
            )
        ).info
        assert info.max_budget == 42.0, f"the budget update itself did not land: {info}"
        servers = (info.object_permission.mcp_servers or []) if info.object_permission else []
        assert servers == [server_id], (
            f"a max_budget-only /key/update dropped the key's MCP servers "
            f"(PR #34452 regression): {info.object_permission}"
        )

        # The stored row surviving is not the thing the customer noticed; they noticed a key
        # that still authenticated but listed no tools. Assert the grant still resolves.
        tools_after = unwrap(client.list_tools(key)).tool_names_for_server(server_id)
        assert tool_before in tools_after, (
            f"the key stopped seeing {tool_before!r} after a budget-only update; the stored "
            f"grant survived but tool discovery did not: {tools_after}"
        )

    @pytest.mark.covers("mcp.key_update.api_key.stale_server_ids_dropped")
    def test_update_with_deleted_server_ids_succeeds_and_drops_them(
        self, client: McpClient, resources: ResourceManager
    ) -> None:
        surviving = register_datadog_mcp(client, resources)
        doomed = register_datadog_mcp(client, resources)
        client.await_registered(surviving)
        client.await_registered(doomed)
        key = _mcp_key(client, resources, [surviving, doomed])

        client.delete_server(doomed)

        # The UI resends the key's full server list, deleted ids included.
        unwrap(
            client.proxy.transport.post(
                "/key/update",
                headers=client.proxy.transport.master,
                json=KeyUpdateBody(
                    key=key,
                    object_permission=ObjectPermission(mcp_servers=[surviving, doomed]),
                ),
                response_type=NoBody,
            )
        )

        assert _key_mcp_servers(client, key) == [surviving], (
            f"/key/update must drop the deleted server id {doomed} and keep {surviving}; "
            f"rejecting or persisting stale ids locks the key out of UI edits (LIT-3278)"
        )
