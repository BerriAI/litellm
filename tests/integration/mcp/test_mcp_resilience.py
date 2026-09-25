import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.mcp import (
    ENTRY_POINTS,
    EntryPoint,
    McpCaller,
    Outcome,
    disconnecting_tool,
    echo_tool,
    listed_tools,
    mcp_peer,
    register_mcp,
    scripted_peer,
    slow_tool,
    tool_calls,
)


def _call(caller: McpCaller, name: str, arguments: dict[str, object], entry: EntryPoint, identity: str) -> Outcome:
    return caller.call(name, arguments, identity if entry == "rest" else None)


def _health(gateway: Gateway, key: str, identity: str) -> str:
    response: Final = gateway.client.get(
        "/v1/mcp/server/health", headers={"x-litellm-api-key": key}, params={"server_ids": [identity]}
    )
    assert response.status_code == 200, response.text
    statuses: Final = {entry["server_id"]: entry["status"] for entry in response.json()}
    assert identity in statuses, response.text
    return str(statuses[identity])


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_tool_error_surfaces_as_error_with_the_peer_message_and_never_as_success(
    gateway: Gateway, entry: EntryPoint
) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "toolerr" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(gateway, key, entry, alias)
        peer.drain()
        outcome: Final = _call(caller, f"{alias}-fail", {}, entry, identity)
        assert outcome.error is not None, f"failing tool reported success: {outcome.raw}"
        assert "Error executing tool fail" in str(outcome.raw), outcome.raw
        assert len(tool_calls(peer.drain())) == 1
        recovered: Final = _call(caller, f"{alias}-add", {"a": 2, "b": 3}, entry, identity)
        assert recovered.text == "5", recovered.raw


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_unreachable_peer_errors_while_a_healthy_sibling_keeps_serving(gateway: Gateway, entry: EntryPoint) -> None:
    with mcp_peer() as healthy, gateway.scenario() as scenario:
        good: Final = "good" + uuid.uuid4().hex[:8]
        bad: Final = "bad" + uuid.uuid4().hex[:8]
        good_id: Final = register_mcp(scenario, healthy, good)
        bad_id: Final = register_mcp(scenario, healthy, bad, url="http://127.0.0.1:9/mcp")
        key: Final = scenario.key(object_permission={"mcp_servers": [good_id, bad_id]})
        caller: Final = McpCaller(gateway, key, entry, good)
        listing: Final = caller.list_tools(good_id if entry == "rest" else None)
        assert listing.error is None, listing.raw
        assert {f"{good}-add", "add"} & set(listing.tools), listing.raw
        assert not [tool for tool in listing.tools if tool.startswith(bad)], listing.raw
        healthy.drain()
        served: Final = _call(caller, f"{good}-add", {"a": 2, "b": 3}, entry, good_id)
        assert served.text == "5", served.raw
        assert len(tool_calls(healthy.drain())) == 1
        if entry == "server_mcp":
            return
        failed: Final = _call(McpCaller(gateway, key, entry, bad), f"{bad}-add", {"a": 2, "b": 3}, entry, bad_id)
        assert failed.error is not None, f"call to unreachable peer succeeded: {failed.raw}"
        assert failed.text != "5"


def test_unreachable_peer_is_reported_unhealthy_and_healthy_peer_healthy(gateway: Gateway) -> None:
    with mcp_peer() as healthy, gateway.scenario() as scenario:
        good: Final = "hgood" + uuid.uuid4().hex[:8]
        bad: Final = "hbad" + uuid.uuid4().hex[:8]
        good_id: Final = register_mcp(scenario, healthy, good)
        bad_id: Final = register_mcp(scenario, healthy, bad, url="http://127.0.0.1:9/mcp")
        assert _health(gateway, gateway.key, good_id) == "healthy"
        assert _health(gateway, gateway.key, bad_id) == "unhealthy"


def test_slow_peer_beyond_configured_timeout_errors_and_does_not_hang_the_gateway(gateway: Gateway) -> None:
    with scripted_peer(slow_tool("nap", 4), echo_tool("echo")) as peer, gateway.scenario() as scenario:
        alias: Final = "slow" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias, timeout=1)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(gateway, key, "mcp", alias)
        peer.drain()
        outcome: Final = caller.call(f"{alias}-nap", {})
        assert outcome.error is not None, f"call past the timeout succeeded: {outcome.raw}"
        assert outcome.text != "slept"
        quick: Final = caller.call(f"{alias}-echo", {"k": "v"})
        assert quick.text == '{"k": "v"}', quick.raw


def test_peer_disconnecting_mid_response_errors_and_the_next_call_succeeds(gateway: Gateway) -> None:
    with scripted_peer(disconnecting_tool("drop"), echo_tool("echo")) as peer, gateway.scenario() as scenario:
        alias: Final = "drop" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        for entry in ("mcp", "rest"):
            caller = McpCaller(gateway, key, entry, alias)
            dropped = caller.call(f"{alias}-drop", {}, identity if entry == "rest" else None)
            assert dropped.error is not None, f"half-written reply became success on {entry}: {dropped.raw}"
            recovered = caller.call(f"{alias}-echo", {"n": 1}, identity if entry == "rest" else None)
            assert recovered.text == '{"n": 1}', recovered.raw


def test_peer_restart_on_the_same_url_is_picked_up_without_gateway_restart(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        alias: Final = "restart" + uuid.uuid4().hex[:8]
        with mcp_peer() as first:
            identity: Final = register_mcp(scenario, first, alias)
            key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
            assert set(listed_tools(gateway, key, identity)) == {"add", "multiply", "fail"}
        caller: Final = McpCaller(gateway, key, "mcp", alias)
        down: Final = caller.call(f"{alias}-add", {"a": 1, "b": 1})
        assert down.error is not None, down.raw
        with scripted_peer(echo_tool("add")) as replacement:
            edited: Final = gateway.request(
                "PUT",
                "/v1/mcp/server",
                {"server_id": identity, "server_name": alias, "alias": alias, **replacement.registration()},
            )
            assert edited.status_code in (200, 202), edited.text
            back: Final = eventually(
                lambda: caller.call(f"{alias}-add", {"a": 1, "b": 1}), lambda outcome: outcome.error is None, seconds=40
            )
            assert back.text == '{"a": 1, "b": 1}', back.raw
            assert len(tool_calls(replacement.drain())) >= 1
