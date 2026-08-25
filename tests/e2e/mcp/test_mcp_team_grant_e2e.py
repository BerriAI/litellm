"""Live e2e: a team MCP grant must not narrow a key's own toolset grant (ticket #7578).

A coding-agent key scoped to exactly one toolset, in a team that holds no MCP grants of
its own, served that toolset's tools fine. Granting the team an unrelated MCP access
group took the same key's discovery to zero, and removing the group restored it, which
is how the customer isolated it during a partial outage.

The cause is the key/team ceiling in `MCPRequestHandler.get_allowed_mcp_servers`:

    base = key_set & team_set  # both restrict -> intersect

A toolset grant resolves into the key's server set and is, in that function's own words,
"subject to the same team/org ceilings as any other key-level grant". While the team
grants nothing, `team_set` is empty and the key keeps its server. The moment the team
gains any grant, `team_set` becomes that grant's servers and intersects away every server
the key reached through its toolset but the team's group does not happen to contain.

So the regression needs a team ceiling that is non-empty and disjoint from the key's
toolset: two Datadog registrations, one carried by the toolset and one reachable only
through the access group handed to the team.

The contract this pins: a grant is a grant. Adding one to a team may widen what its keys
can reach and must never remove a tool a key could already see.
"""

from __future__ import annotations

import pytest
from datadog_mcp import SEARCH_LOGS_TOOL, register_datadog_mcp
from e2e_config import unique_marker
from e2e_http import NoBody, unwrap
from lifecycle import ResourceManager
from mcp_client import McpClient
from models import (
    KeyGenerateBody,
    ObjectPermission,
    TeamNewBody,
    TeamNewResponse,
    TeamUpdateBody,
)
from pydantic import BaseModel

pytestmark = pytest.mark.e2e


class ToolsetTool(BaseModel):
    server_id: str
    tool_name: str


class ToolsetCreateBody(BaseModel):
    toolset_name: str
    tools: list[ToolsetTool]


class Toolset(BaseModel):
    toolset_id: str


class TeamDeleteBody(BaseModel):
    team_ids: list[str]


def _create_toolset(client: McpClient, resources: ResourceManager, *, server_id: str) -> str:
    toolset = unwrap(
        client.proxy.transport.post(
            "/v1/mcp/toolset",
            headers=client.proxy.transport.master,
            json=ToolsetCreateBody(
                toolset_name=f"e2e-toolset-{unique_marker()}",
                tools=[ToolsetTool(server_id=server_id, tool_name=SEARCH_LOGS_TOOL)],
            ),
            response_type=Toolset,
        )
    )
    resources.defer(
        lambda: client.proxy.transport.delete(
            f"/v1/mcp/toolset/{toolset.toolset_id}",
            headers=client.proxy.transport.master,
            json=NoBody(),
            response_type=NoBody,
        )
    )
    return toolset.toolset_id


def _create_team(client: McpClient, resources: ResourceManager) -> str:
    team = unwrap(
        client.proxy.transport.post(
            "/team/new",
            headers=client.proxy.transport.master,
            json=TeamNewBody(team_alias=f"e2e-mcp-team-{unique_marker()}"),
            response_type=TeamNewResponse,
        )
    )
    resources.defer(
        lambda: client.proxy.transport.post(
            "/team/delete",
            headers=client.proxy.transport.master,
            json=TeamDeleteBody(team_ids=[team.team_id]),
            response_type=NoBody,
        )
    )
    return team.team_id


def _grant_team_access_group(client: McpClient, team_id: str, access_group: str) -> None:
    _ = unwrap(
        client.proxy.transport.post(
            "/team/update",
            headers=client.proxy.transport.master,
            json=TeamUpdateBody(
                team_id=team_id,
                object_permission=ObjectPermission(mcp_access_groups=[access_group]),
            ),
            response_type=NoBody,
        )
    )


@pytest.mark.skip(
    reason="ticket #7578: open product bug. get_allowed_mcp_servers intersects the key's "
    "toolset-resolved servers with the team's ceiling (user_api_key_auth_mcp.py:1443 "
    "'base = key_set & team_set'), so granting a team an unrelated MCP access group "
    "erases a child key's toolset tools. Unskip when the fix lands."
)
class TestTeamGrantDoesNotNarrowKeyToolset:
    @pytest.mark.covers("mcp.list_tools.api_key.team_grant_preserves_key_toolset")
    def test_adding_a_team_access_group_keeps_the_key_toolset_tools(
        self, client: McpClient, resources: ResourceManager
    ) -> None:
        access_group = f"e2e_ag_{unique_marker()}"
        toolset_server = register_datadog_mcp(client, resources)
        group_server = register_datadog_mcp(client, resources, mcp_access_groups=[access_group])
        client.await_registered(toolset_server)
        client.await_registered(group_server)

        toolset_id = _create_toolset(client, resources, server_id=toolset_server)
        team_id = _create_team(client, resources)
        key = client.proxy.generate_key(
            KeyGenerateBody(
                user_id=f"e2e-team-grant-{unique_marker()}",
                team_id=team_id,
                object_permission=ObjectPermission(mcp_toolsets=[toolset_id]),
            )
        )
        resources.defer(lambda: client.proxy.delete_key(key))

        before = client.await_tool(key, toolset_server, SEARCH_LOGS_TOOL)

        _grant_team_access_group(client, team_id, access_group)

        after = unwrap(client.list_tools(key)).tool_names_for_server(toolset_server)
        assert before in after, (
            f"granting team {team_id} the unrelated MCP access group {access_group!r} removed "
            f"{before!r} from a key whose own grant is toolset {toolset_id} (ticket #7578: this "
            f"took a customer's coding agents down until they removed the group). A team grant "
            f"must widen access, never narrow what the key already reached; the key now sees "
            f"{sorted(after)}"
        )
