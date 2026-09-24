import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from typing import Final

import pytest
from integration._support.client import Gateway, JsonValue, Scenario, eventually
from integration._support.database import read_rows
from integration._support.mcp import (
    ENTRY_POINTS,
    EntryPoint,
    McpCaller,
    McpPeer,
    Outcome,
    mcp_peer,
    register_mcp,
    tool_calls,
)

DEFAULT_COST: Final = 0.25
ADD_COST: Final = 0.5
FORBIDDEN: Final = "forbidden-integration-word"
SPEND_ROWS: Final = (
    'SELECT call_type, model, spend, status, metadata FROM "LiteLLM_SpendLogs" WHERE api_key = %s ORDER BY "startTime"'
)


def _digest(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def _rows(key: str, count: int) -> list[dict[str, JsonValue]]:
    return eventually(lambda: read_rows(SPEND_ROWS, (_digest(key),)), lambda rows: len(rows) >= count, seconds=70)


def _priced_server(scenario: Scenario, peer: McpPeer, alias: str) -> str:
    return register_mcp(
        scenario,
        peer,
        alias,
        mcp_info={
            "server_name": alias,
            "mcp_server_cost_info": {
                "default_cost_per_query": DEFAULT_COST,
                "tool_name_to_cost_per_query": {"add": ADD_COST},
            },
        },
    )


def _tool_metadata(row: dict[str, JsonValue]) -> dict[str, JsonValue]:
    metadata: Final = row["metadata"]
    assert isinstance(metadata, dict), row
    tool: Final = metadata.get("mcp_tool_call_metadata")
    assert isinstance(tool, dict), metadata
    return tool


def _call(caller: McpCaller, name: str, arguments: dict[str, object], entry: EntryPoint, identity: str) -> Outcome:
    return caller.call(name, arguments, identity if entry == "rest" else None)


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_each_tool_call_writes_one_spend_row_with_server_tool_and_configured_cost(
    gateway: Gateway, entry: EntryPoint
) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "acct" + uuid.uuid4().hex[:8]
        identity: Final = _priced_server(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(gateway, key, entry, alias)
        peer.drain()
        assert _call(caller, f"{alias}-add", {"a": 2, "b": 3}, entry, identity).text == "5"
        assert _call(caller, f"{alias}-multiply", {"a": 2, "b": 3}, entry, identity).text == "6"
        assert len(tool_calls(peer.drain())) == 2
        rows: Final = [row for row in _rows(key, 2) if row["call_type"] == "call_mcp_tool"]
        assert len(rows) == 2, rows
        by_tool: Final = {_tool_metadata(row)["name"]: row for row in rows}
        assert set(by_tool) == {"add", "multiply"}, rows
        assert float(str(by_tool["add"]["spend"])) == pytest.approx(ADD_COST)
        assert float(str(by_tool["multiply"]["spend"])) == pytest.approx(DEFAULT_COST)
        for row in rows:
            assert _tool_metadata(row)["mcp_server_name"] == alias, row
            assert row["model"] == f"MCP: {alias}-{_tool_metadata(row)['name']}", row
        later: Final = read_rows(SPEND_ROWS, (_digest(key),))
        assert len([row for row in later if row["call_type"] == "call_mcp_tool"]) == 2, later


def test_key_spend_and_key_max_budget_count_mcp_tool_calls(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "budget" + uuid.uuid4().hex[:8]
        identity: Final = _priced_server(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]}, max_budget=ADD_COST / 2)
        caller: Final = McpCaller(gateway, key, "mcp", alias)
        assert caller.call(f"{alias}-add", {"a": 2, "b": 3}).text == "5"
        info: Final = eventually(
            lambda: gateway.client.get("/key/info", params={"key": key}, headers={"x-litellm-api-key": gateway.key}),
            lambda response: response.status_code == 200 and float(response.json()["info"]["spend"]) > 0,
            seconds=70,
        )
        assert float(info.json()["info"]["spend"]) == pytest.approx(ADD_COST)
        eventually(
            lambda: caller.call(f"{alias}-add", {"a": 2, "b": 3}),
            lambda outcome: outcome.error is not None,
            seconds=70,
        )
        peer.drain()
        denied: Final = caller.call(f"{alias}-add", {"a": 2, "b": 3})
        assert denied.error is not None and "budget" in str(denied.raw).lower(), denied.raw
        assert tool_calls(peer.drain()) == (), "over-budget call reached the peer"


@contextmanager
def _content_filter(gateway: Gateway, mode: str) -> Iterator[str]:
    name: Final = "filter" + uuid.uuid4().hex[:8]
    created: Final = gateway.client.post(
        "/guardrails",
        headers={"x-litellm-api-key": gateway.key},
        json={
            "guardrail": {
                "guardrail_name": name,
                "litellm_params": {
                    "guardrail": "litellm_content_filter",
                    "mode": mode,
                    "default_on": True,
                    "blocked_words": [{"keyword": FORBIDDEN, "action": "BLOCK"}],
                },
            }
        },
    )
    assert created.status_code == 200, created.text
    identity: Final = created.json()["guardrail_id"]
    try:
        yield name
    finally:
        deleted: Final = gateway.client.delete(f"/guardrails/{identity}", headers={"x-litellm-api-key": gateway.key})
        assert deleted.status_code == 200, deleted.text


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_pre_mcp_call_guardrail_blocks_before_the_peer_and_still_logs_spend(
    gateway: Gateway, entry: EntryPoint
) -> None:
    with _content_filter(gateway, "pre_mcp_call"), mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "guard" + uuid.uuid4().hex[:8]
        identity: Final = _priced_server(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(gateway, key, entry, alias)
        peer.drain()
        clean: Final = _call(caller, f"{alias}-add", {"a": 2, "b": 3}, entry, identity)
        assert clean.text == "5", clean.raw
        blocked: Final = _call(caller, f"{alias}-add", {"a": 1, "b": FORBIDDEN}, entry, identity)
        assert blocked.error is not None, f"guardrail-blocked call succeeded: {blocked.raw}"
        assert FORBIDDEN in str(blocked.raw) or "blocked" in str(blocked.raw).lower(), blocked.raw
        assert len(tool_calls(peer.drain())) == 1, "blocked call reached the peer"
        rows: Final = [row for row in _rows(key, 2) if row["call_type"] == "call_mcp_tool"]
        assert len(rows) == 2, rows
        failures: Final = [row for row in rows if row["status"] == "failure"]
        assert len(failures) == 1, rows
        assert failures[0]["model"] == f"MCP: {alias}-add", failures[0]
        assert _tool_metadata(failures[0])["mcp_server_name"] == alias, failures[0]


def test_guardrail_blocked_call_never_reaches_peer_through_the_official_client(gateway: Gateway) -> None:
    with _content_filter(gateway, "pre_mcp_call"), mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "guardsdk" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(gateway, key, "server_mcp", alias)
        peer.drain()
        blocked: Final = caller.call(f"{alias}-add", {"a": 1, "b": FORBIDDEN})
        assert blocked.error is not None, blocked.raw
        assert tool_calls(peer.drain()) == ()
        allowed: Final = caller.call(f"{alias}-add", {"a": 4, "b": 5})
        assert allowed.text == "9", allowed.raw
        assert len(tool_calls(peer.drain())) == 1


def test_guardrail_removal_stops_blocking_without_restart(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "guardoff" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(gateway, key, "mcp", alias)
        with _content_filter(gateway, "pre_mcp_call"):
            assert caller.call(f"{alias}-add", {"a": 1, "b": FORBIDDEN}).error is not None
        peer.drain()
        eventually(
            lambda: (caller.call(f"{alias}-add", {"a": 1, "b": FORBIDDEN}), tool_calls(peer.drain()))[1],
            lambda calls: len(calls) >= 1,
            seconds=40,
        )
