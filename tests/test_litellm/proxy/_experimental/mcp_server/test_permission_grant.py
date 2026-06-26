import dataclasses

import pytest

from litellm.proxy._experimental.mcp_server.permission_grant import (
    AllServers,
    AllTeamServers,
    ExplicitServers,
    MCPServerGrant,
    NoServers,
    parse_mcp_server_grant,
)
from litellm.proxy._types import SpecialMCPServerNames

NO_MCP = SpecialMCPServerNames.no_mcp_servers.value
ALL_PROXY = SpecialMCPServerNames.all_proxy_mcp_servers.value
ALL_TEAM = SpecialMCPServerNames.all_team_mcp_servers.value


def test_concrete_only_returns_explicit_servers() -> None:
    grant = parse_mcp_server_grant(["srv-a", "srv-b", "alias-c"])
    assert grant == ExplicitServers(frozenset({"srv-a", "srv-b", "alias-c"}))


def test_all_proxy_sentinel_returns_all_servers() -> None:
    assert parse_mcp_server_grant([ALL_PROXY]) == AllServers()


def test_all_proxy_sentinel_dominates_concrete_ids() -> None:
    assert parse_mcp_server_grant([ALL_PROXY, "srv-a"]) == AllServers()


def test_no_mcp_sentinel_returns_no_servers() -> None:
    assert parse_mcp_server_grant([NO_MCP]) == NoServers()


def test_all_team_sentinel_returns_all_team_servers() -> None:
    assert parse_mcp_server_grant([ALL_TEAM]) == AllTeamServers()


def test_all_team_sentinel_dominates_concrete_ids() -> None:
    assert parse_mcp_server_grant([ALL_TEAM, "srv-a"]) == AllTeamServers()


def test_block_all_wins_over_all_proxy_regardless_of_order() -> None:
    assert parse_mcp_server_grant([NO_MCP, ALL_PROXY]) == NoServers()
    assert parse_mcp_server_grant([ALL_PROXY, NO_MCP]) == NoServers()


def test_block_all_wins_over_all_team_regardless_of_order() -> None:
    assert parse_mcp_server_grant([NO_MCP, ALL_TEAM]) == NoServers()
    assert parse_mcp_server_grant([ALL_TEAM, NO_MCP]) == NoServers()


def test_all_team_beats_all_proxy_regardless_of_order() -> None:
    # all-team caps to the team and so is the more restrictive grant; it wins
    # over the broader all-proxy when both somehow appear.
    assert parse_mcp_server_grant([ALL_PROXY, ALL_TEAM]) == AllTeamServers()
    assert parse_mcp_server_grant([ALL_TEAM, ALL_PROXY]) == AllTeamServers()


def test_empty_list_returns_empty_explicit_servers() -> None:
    assert parse_mcp_server_grant([]) == ExplicitServers(frozenset())


def test_grants_are_frozen() -> None:
    grant = ExplicitServers(frozenset({"srv-a"}))
    with pytest.raises(dataclasses.FrozenInstanceError):
        grant.identifiers = frozenset({"other"})  # pyright: ignore[reportAttributeAccessIssue]


def test_grants_equal_by_value() -> None:
    assert AllServers() == AllServers()
    assert AllTeamServers() == AllTeamServers()
    assert NoServers() == NoServers()
    assert ExplicitServers(frozenset({"a"})) == ExplicitServers(frozenset({"a"}))
    assert AllServers() != NoServers()
    assert AllServers() != AllTeamServers()


def test_union_alias_admits_each_variant() -> None:
    grants: tuple[MCPServerGrant, ...] = (
        AllServers(),
        AllTeamServers(),
        NoServers(),
        ExplicitServers(frozenset({"a"})),
    )
    assert len(grants) == 4
