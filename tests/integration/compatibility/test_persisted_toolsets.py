import json
import os
import uuid
from pathlib import Path
from typing import Final

import psycopg
import pytest

from integration._support.client import Gateway
from integration._support.database import read_rows
from integration._support.mcp import call_tool, mcp_peer, register_mcp, tool_names
from integration._support.process import owned_proxy


@pytest.mark.covers("other.compatibility.mcp.persisted_tool_names_survive_candidate_startup")
def test_existing_toolset_format_loads_before_start_and_keeps_sibling_denied(gateway: Gateway, tmp_path: Path) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        identity: Final = register_mcp(scenario, peer, "integration" + uuid.uuid4().hex)
        toolset: Final = str(uuid.uuid4())

        def cleanup() -> None:
            response: Final = gateway.request("DELETE", f"/v1/mcp/toolset/{toolset}")
            assert response.status_code == 202, response.text
            assert read_rows('SELECT toolset_id FROM "LiteLLM_MCPToolsetTable" WHERE toolset_id=%s', (toolset,)) == []

        with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
            connection.execute(
                'INSERT INTO "LiteLLM_MCPToolsetTable" (toolset_id, toolset_name, tools, updated_at) '
                'VALUES (%s,%s,%s::jsonb,NOW())',
                (toolset, "integration" + uuid.uuid4().hex, json.dumps([{"server_id": identity, "tool_name": "add"}])),
            )
        scenario.cleanups.callback(cleanup)
        key: Final = scenario.key(object_permission={"mcp_toolsets": [toolset]})
        control: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        with owned_proxy(gateway, tmp_path, {}) as candidate:
            full: Final = tool_names(candidate, control, identity)
            names: Final = tool_names(candidate, key, identity)
            assert set(names) == {"add"} and set(full) == {"add", "multiply", "fail"}
            result: Final = call_tool(candidate, key, identity, names["add"], {"a": 3, "b": 5})
            assert result.status_code == 200 and result.json()["isError"] is False, result.text
            assert result.json()["content"][0]["text"] == "8"
            peer.drain()
            denied: Final = call_tool(candidate, key, identity, full["multiply"], {"a": 3, "b": 5})
            assert denied.status_code == 403, denied.text
            assert "access" in denied.text.lower()
            assert not tuple(item for item in peer.drain() if item["body"].get("method") == "tools/call")
            result: Final = call_tool(candidate, control, identity, full["multiply"], {"a": 3, "b": 5})
            assert result.status_code == 200 and result.json()["isError"] is False, result.text
            assert result.json()["content"][0]["text"] == "15"
