import uuid
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.mcp import (
    ENTRY_POINTS,
    EntryPoint,
    McpCaller,
    McpPeer,
    PeerKind,
    peer_of,
    register_mcp,
    tool_calls,
)
from integration._support.mcp_grants import SUBJECTS, Subject, grant

CALLABLE: Final = {"add": {"a": 1, "b": 2}, "multiply": {"a": 2, "b": 3}}
RESULTS: Final = {"add": "3", "multiply": "6"}


def _server_scoped(entry: EntryPoint, identity: str) -> str | None:
    return identity if entry == "rest" else None


def _name(entry: EntryPoint, alias: str, tool: str) -> str:
    return tool if entry == "rest" else f"{alias}-{tool}"


def _assert_denied(caller: McpCaller, peer: McpPeer, name: str, identity: str, entry: EntryPoint) -> None:
    peer.drain()
    outcome: Final = caller.call(name, CALLABLE["add"], _server_scoped(entry, identity))
    assert outcome.error is not None, f"denied call succeeded: {outcome.raw}"
    assert outcome.text not in RESULTS.values(), outcome.raw
    assert tool_calls(peer.drain()) == (), "denied call reached the peer"


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("subject", SUBJECTS)
@pytest.mark.parametrize("peer_kind", ("http", "sse"))
def test_subject_grant_lists_only_reachable_tools_and_denies_the_rest(
    gateway: Gateway, peer_kind: PeerKind, subject: Subject, entry: EntryPoint
) -> None:
    with peer_of(peer_kind) as granted_peer, peer_of(peer_kind) as denied_peer, gateway.scenario() as scenario:
        group: Final = "grp" + uuid.uuid4().hex[:8]
        granted_alias: Final = "yes" + uuid.uuid4().hex[:8]
        denied_alias: Final = "no" + uuid.uuid4().hex[:8]
        granted: Final = register_mcp(scenario, granted_peer, granted_alias, mcp_access_groups=[group])
        denied: Final = register_mcp(scenario, denied_peer, denied_alias)
        caller: Final = grant(
            scenario, subject, (granted,), (granted, denied), access_group=group, allowed_tools={granted: ("add",)}
        )
        reach: Final = McpCaller(gateway, caller.key, entry, granted_alias, caller.headers)
        listed: Final = reach.list_tools(_server_scoped(entry, granted))
        assert listed.ok, listed.raw
        expected: Final = (
            {_name(entry, granted_alias, "add")}
            if subject in ("toolset", "allowed_tools")
            else {_name(entry, granted_alias, tool) for tool in ("add", "multiply", "fail")}
        )
        assert set(listed.tools) == expected, listed.tools
        for tool, arguments in CALLABLE.items():
            name: Final = _name(entry, granted_alias, tool)
            if name not in listed.tools:
                continue
            granted_peer.drain()
            outcome: Final = reach.call(name, arguments, _server_scoped(entry, granted))
            assert outcome.ok and outcome.text == RESULTS[tool], outcome.raw
            assert [call["body"]["params"]["name"] for call in tool_calls(granted_peer.drain())] == [tool]
        if subject in ("toolset", "allowed_tools"):
            _assert_denied(reach, granted_peer, _name(entry, granted_alias, "multiply"), granted, entry)
        blocked: Final = McpCaller(gateway, caller.key, entry, denied_alias, caller.headers)
        _assert_denied(blocked, denied_peer, _name(entry, denied_alias, "add"), denied, entry)
        denied_listed: Final = blocked.list_tools(_server_scoped(entry, denied))
        if entry == "rest":
            assert denied_listed.status == 403 and "access_denied" in denied_listed.raw, denied_listed.raw
            assert denied_listed.tools == ()
        else:
            assert not any(name.startswith(denied_alias) for name in denied_listed.tools), denied_listed.tools


@pytest.mark.parametrize("entry", ("mcp", "server_mcp", "rest"))
def test_key_without_any_grant_sees_no_scoped_server(gateway: Gateway, entry: EntryPoint) -> None:
    with peer_of("http") as peer, gateway.scenario() as scenario:
        alias: Final = "none" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        other: Final = scenario.key(object_permission={"mcp_servers": ["no-mcp-servers"]})
        caller: Final = McpCaller(gateway, other, entry, alias)
        _assert_denied(caller, peer, _name(entry, alias, "add"), identity, entry)
        listed: Final = caller.list_tools(_server_scoped(entry, identity))
        assert not any(name.startswith(alias) for name in listed.tools), listed.tools


@pytest.mark.parametrize("entry", ("mcp", "server_mcp", "rest", "root", "sse"))
def test_missing_or_wrong_key_is_rejected_before_the_peer(gateway: Gateway, entry: EntryPoint) -> None:
    with peer_of("http") as peer, gateway.scenario() as scenario:
        alias: Final = "anon" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        for key in (None, "sk-integration-wrong-" + uuid.uuid4().hex):
            caller: Final = McpCaller(gateway, key, entry, alias)
            peer.drain()
            outcome: Final = caller.call(_name(entry, alias, "add"), CALLABLE["add"], _server_scoped(entry, identity))
            assert outcome.status in (401, 403) or outcome.error is not None, outcome.raw
            assert outcome.text not in RESULTS.values(), outcome.raw
            assert tool_calls(peer.drain()) == ()


def test_same_tool_name_on_two_servers_routes_by_prefix(gateway: Gateway) -> None:
    with peer_of("http") as first, peer_of("sse") as second, gateway.scenario() as scenario:
        first_alias: Final = "one" + uuid.uuid4().hex[:8]
        second_alias: Final = "two" + uuid.uuid4().hex[:8]
        first_id: Final = register_mcp(scenario, first, first_alias)
        second_id: Final = register_mcp(scenario, second, second_alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [first_id, second_id]})
        caller: Final = McpCaller(gateway, key, "mcp", None)
        listed: Final = caller.list_tools()
        assert listed.ok and len(listed.tools) == len(set(listed.tools)) == 6, listed.tools
        assert {f"{first_alias}-add", f"{second_alias}-add"} <= set(listed.tools)
        first.drain()
        second.drain()
        outcome: Final = caller.call(f"{second_alias}-add", {"a": 5, "b": 5})
        assert outcome.ok and outcome.text == "10", outcome.raw
        assert tool_calls(first.drain()) == ()
        assert [call["body"]["params"]["name"] for call in tool_calls(second.drain())] == ["add"]
