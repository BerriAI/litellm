import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Literal

from integration._support.client import Gateway, Scenario, string_value

Subject = Literal["key", "team", "org", "user", "end_user", "agent", "access_group", "toolset", "allowed_tools"]
SUBJECTS: Final[tuple[Subject, ...]] = (
    "key",
    "team",
    "org",
    "user",
    "end_user",
    "agent",
    "access_group",
    "toolset",
    "allowed_tools",
)


@dataclass(frozen=True, slots=True)
class Caller:
    """A key plus the request headers that make the proxy resolve the granted subject."""

    key: str
    headers: Mapping[str, str]


def _mcp_permission(server_ids: tuple[str, ...]) -> dict[str, list[str]]:
    return {"mcp_servers": list(server_ids)}


def delete_organization(gateway: Gateway, identity: str) -> None:
    response: Final = gateway.request("DELETE", "/organization/delete", {"organization_ids": [identity]})
    assert response.status_code == 200, response.text


def delete_end_user(gateway: Gateway, identity: str) -> None:
    response: Final = gateway.request("POST", "/end_user/delete", {"user_ids": [identity]})
    assert response.status_code == 200, response.text


def delete_agent(gateway: Gateway, identity: str) -> None:
    response: Final = gateway.request("DELETE", f"/v1/agents/{identity}")
    assert response.status_code == 200, response.text


def delete_toolset(gateway: Gateway, identity: str) -> None:
    response: Final = gateway.request("DELETE", f"/v1/mcp/toolset/{identity}")
    assert response.status_code in (200, 202, 204), response.text


def create_toolset(scenario: Scenario, tools: tuple[tuple[str, str], ...]) -> str:
    response: Final = scenario.gateway.request(
        "POST",
        "/v1/mcp/toolset",
        {
            "toolset_name": f"integration-{uuid.uuid4().hex[:10]}",
            "tools": [{"server_id": server_id, "tool_name": tool} for server_id, tool in tools],
        },
    )
    assert response.status_code == 201, response.text
    identity: Final = string_value(response.json()["toolset_id"])
    scenario.cleanups.callback(delete_toolset, scenario.gateway, identity)
    return identity


def grant(
    scenario: Scenario,
    subject: Subject,
    granted: tuple[str, ...],
    ceiling: tuple[str, ...],
    *,
    access_group: str | None = None,
    allowed_tools: Mapping[str, tuple[str, ...]] | None = None,
) -> Caller:
    """Build a caller whose ``subject`` level grants exactly ``granted`` out of ``ceiling``.

    ``ceiling`` is what the key itself can reach before the subject narrows it; the key subject grants
    ``granted`` directly. Access groups take the group name that the granted servers were registered with,
    and ``allowed_tools`` maps server id to the tools the key may call on it."""
    gateway: Final = scenario.gateway
    match subject:
        case "key":
            return Caller(scenario.key(object_permission=_mcp_permission(granted)), {})
        case "team":
            team: Final = scenario.team(object_permission=_mcp_permission(granted))
            return Caller(scenario.key(team_id=team), {})
        case "org":
            created: Final = gateway.post(
                "/organization/new",
                {
                    "organization_alias": f"integration-{uuid.uuid4().hex[:10]}",
                    "object_permission": _mcp_permission(granted),
                },
            )
            org: Final = string_value(created["organization_id"])
            scenario.cleanups.callback(delete_organization, gateway, org)
            org_team: Final = scenario.team(organization_id=org, object_permission=_mcp_permission(ceiling))
            return Caller(scenario.key(team_id=org_team), {})
        case "user":
            user: Final = scenario.user(object_permission=_mcp_permission(granted))
            return Caller(scenario.key(user_id=user, object_permission=_mcp_permission(ceiling)), {})
        case "end_user":
            end_user: Final = f"integration-{uuid.uuid4().hex[:10]}"
            response: Final = gateway.request(
                "POST", "/end_user/new", {"user_id": end_user, "object_permission": _mcp_permission(granted)}
            )
            assert response.status_code == 200, response.text
            scenario.cleanups.callback(delete_end_user, gateway, end_user)
            return Caller(scenario.key(object_permission=_mcp_permission(ceiling)), {"x-litellm-end-user-id": end_user})
        case "agent":
            agent: Final = gateway.post(
                "/v1/agents",
                {
                    "agent_name": f"integration-{uuid.uuid4().hex[:10]}",
                    "agent_card_params": {
                        "protocolVersion": "0.3.0",
                        "name": "integration",
                        "description": "integration agent",
                        "url": "http://127.0.0.1:1/agent",
                        "version": "1",
                        "capabilities": {},
                        "defaultInputModes": ["text"],
                        "defaultOutputModes": ["text"],
                        "skills": [],
                    },
                    "object_permission": _mcp_permission(granted),
                },
            )
            agent_id: Final = string_value(agent["agent_id"])
            scenario.cleanups.callback(delete_agent, gateway, agent_id)
            return Caller(scenario.key(agent_id=agent_id, object_permission=_mcp_permission(ceiling)), {})
        case "access_group":
            assert access_group is not None
            return Caller(scenario.key(object_permission={"mcp_access_groups": [access_group]}), {})
        case "toolset":
            toolset: Final = create_toolset(scenario, tuple((server, "add") for server in granted))
            return Caller(scenario.key(object_permission={"mcp_toolsets": [toolset]}), {})
        case "allowed_tools":
            assert allowed_tools is not None
            return Caller(
                scenario.key(
                    object_permission={
                        "mcp_servers": list(granted),
                        "mcp_tool_permissions": {server: list(tools) for server, tools in allowed_tools.items()},
                    }
                ),
                {},
            )
