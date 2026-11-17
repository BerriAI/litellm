"""MCP cost and spend rows on the Python gateway.

The contract: every MCP tool call writes one `LiteLLM_SpendLogs` row with `call_type = call_mcp_tool`,
`model = "MCP: <prefixed tool name>"` and `metadata.mcp_tool_call_metadata`. Its spend comes from
`MCPCostCalculator.calculate_mcp_tool_call_cost` (`litellm/proxy/mcp_server/cost_calculator.py`): the
bare tool name looked up in `mcp_info.mcp_server_cost_info.tool_name_to_cost_per_query`, else
`default_cost_per_query`, else 0.0. Config costs are coerced to float at ingest
(`_normalize_mcp_server_cost_info`), so `7e-05` written as a YAML string still prices. A `tools/list`
over the MCP protocol writes a `list_mcp_tools` row (`model = "MCP: list_tools"`); the REST listing
does not. A JSON-RPC request the SDK rejects (unknown method, invalid params) writes one failure row
with `model = "MCP: request_rejected"` (`log_mcp_sdk_rejection` in `operations.py`).

The Rust MCP host has no spend logging and refuses `mcp_server_cost_info` at boot
(`ConfigFault::UnsupportedMcpInfo`), so every row shape here is something the port must rebuild.
`test_mcp_accounting_guardrails.py` already covers the priced per-tool and default cost and key
budget; this file covers the remaining branches.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.mcp import McpCaller, McpPeer, mcp_peer, register_mcp, tool_calls
from integration._support.process import owned_proxy

from tests.integration.mcp.dec1_mcp_support import (
    base_config,
    digest,
    metadata_of,
    rpc,
    rpc_error,
    server_ids_by_name,
    spend_rows,
)

pytestmark: Final = pytest.mark.shares_worker_state


def _priced(scenario, peer: McpPeer, alias: str, cost_info: dict[str, object]) -> str:  # noqa: ANN001
    return register_mcp(scenario, peer, alias, mcp_info={"server_name": alias, "mcp_server_cost_info": cost_info})


def _of_type(rows: list[dict[str, object]], call_type: str) -> list[dict[str, object]]:
    return [row for row in rows if row["call_type"] == call_type]


def test_a_tool_call_on_an_unpriced_server_writes_one_zero_spend_success_row(gateway: Gateway) -> None:
    """A server with no `mcp_info` cost settings. One `<alias>-add` call through `/mcp-rest/tools/call`
    must land exactly one `call_mcp_tool` row with `status = success`, `spend = 0` and
    `model = "MCP: <alias>-add"`. A port that skipped the row for free tools would lose the audit trail."""
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "dec1free" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        assert McpCaller(gateway, key, "rest").call(f"{alias}-add", {"a": 2, "b": 3}, identity).text == "5"
        rows: Final = _of_type(spend_rows(key, 1), "call_mcp_tool")
        assert len(rows) == 1, rows
        assert rows[0]["status"] == "success", rows
        assert float(str(rows[0]["spend"])) == 0.0, rows
        assert rows[0]["model"] == f"MCP: {alias}-add", rows


def test_a_cost_map_keyed_by_the_prefixed_tool_name_does_not_price_the_call(gateway: Gateway) -> None:
    """`tool_name_to_cost_per_query` is `{"<alias>-add": 0.5}` with `default_cost_per_query: 0.1`. The
    calculator looks the cost up by the bare name from `mcp_tool_call_metadata.name` (`add`), so the
    prefixed key never matches and the row must be charged the default 0.1. Pinned so a port does not
    silently start honouring (or rejecting) prefixed keys."""
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "dec1pfx" + uuid.uuid4().hex[:8]
        identity: Final = _priced(
            scenario,
            peer,
            alias,
            {"default_cost_per_query": 0.1, "tool_name_to_cost_per_query": {f"{alias}-add": 0.5}},
        )
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        assert McpCaller(gateway, key, "mcp").call(f"{alias}-add", {"a": 2, "b": 3}).text == "5"
        rows: Final = _of_type(spend_rows(key, 1), "call_mcp_tool")
        assert len(rows) == 1, rows
        assert float(str(rows[0]["spend"])) == pytest.approx(0.1), rows
        assert metadata_of(rows[0])["mcp_tool_call_metadata"]["name"] == "add", rows


def test_a_per_tool_cost_of_zero_overrides_a_nonzero_default(gateway: Gateway) -> None:
    """`tool_name_to_cost_per_query: {add: 0}` with `default_cost_per_query: 0.3`. The per-tool entry wins
    even when it is zero (`if tool_name in tool_name_to_cost_per_query`), so `add` costs 0.0 and
    `multiply` costs 0.3. A port that tested truthiness would charge `add` the default."""
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "dec1zero" + uuid.uuid4().hex[:8]
        identity: Final = _priced(
            scenario, peer, alias, {"default_cost_per_query": 0.3, "tool_name_to_cost_per_query": {"add": 0}}
        )
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(gateway, key, "mcp")
        assert caller.call(f"{alias}-add", {"a": 1, "b": 1}).text == "2"
        assert caller.call(f"{alias}-multiply", {"a": 2, "b": 2}).text == "4"
        rows: Final = _of_type(spend_rows(key, 2), "call_mcp_tool")
        by_tool: Final = {metadata_of(row)["mcp_tool_call_metadata"]["name"]: float(str(row["spend"])) for row in rows}
        assert by_tool == {"add": 0.0, "multiply": pytest.approx(0.3)}, rows


def test_a_yaml_string_cost_in_scientific_notation_is_coerced_and_charged(gateway: Gateway, tmp_path: Path) -> None:
    """A config-declared server has `default_cost_per_query: "7e-05"` (a string, as YAML 1.1 parses
    `7e-05`). `_normalize_mcp_server_cost_info` coerces it to float at load, so one tool call must land a
    row with spend 7e-05. The Rust host refuses any `mcp_server_cost_info` at boot today."""
    with mcp_peer() as peer:
        name: Final = "dec1sci" + uuid.uuid4().hex[:8]
        config: Final = base_config()
        config["mcp_servers"] = {
            name: {
                **peer.registration(),
                "allow_all_keys": True,
                "mcp_info": {"mcp_server_cost_info": {"default_cost_per_query": "7e-05"}},
            }
        }
        path: Final = tmp_path / "sci.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as owned, owned.scenario() as scenario:
            key: Final = scenario.key()
            assert McpCaller(owned, key, "mcp").call(f"{name}-add", {"a": 2, "b": 3}).text == "5"
            rows: Final = eventually(
                lambda: owned.rows(
                    'SELECT spend FROM "LiteLLM_SpendLogs" WHERE api_key = %s AND call_type = %s',
                    (digest(key), "call_mcp_tool"),
                ),
                lambda found: len(found) == 1,
                seconds=70,
            )
            assert float(str(rows[0]["spend"])) == pytest.approx(7e-05), rows


def test_a_non_numeric_config_cost_is_dropped_and_the_server_still_serves(gateway: Gateway, tmp_path: Path) -> None:
    """A config-declared server has `default_cost_per_query: "free"` and `tool_name_to_cost_per_query:
    {add: "lots", multiply: 0.2}`. Bad values are dropped with a warning instead of failing the load, so
    the server is listed, `add` is charged 0.0 and `multiply` 0.2."""
    with mcp_peer() as peer:
        name: Final = "dec1bad" + uuid.uuid4().hex[:8]
        config: Final = base_config()
        config["mcp_servers"] = {
            name: {
                **peer.registration(),
                "allow_all_keys": True,
                "mcp_info": {
                    "mcp_server_cost_info": {
                        "default_cost_per_query": "free",
                        "tool_name_to_cost_per_query": {"add": "lots", "multiply": 0.2},
                    }
                },
            }
        }
        path: Final = tmp_path / "bad-cost.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as owned, owned.scenario() as scenario:
            assert name in server_ids_by_name(owned)
            key: Final = scenario.key()
            caller: Final = McpCaller(owned, key, "mcp")
            assert caller.call(f"{name}-add", {"a": 1, "b": 1}).text == "2"
            assert caller.call(f"{name}-multiply", {"a": 2, "b": 2}).text == "4"
            rows: Final = eventually(
                lambda: owned.rows(
                    'SELECT spend, metadata FROM "LiteLLM_SpendLogs" WHERE api_key = %s AND call_type = %s',
                    (digest(key), "call_mcp_tool"),
                ),
                lambda found: len(found) == 2,
                seconds=70,
            )
            by_tool: Final = {
                metadata_of(row)["mcp_tool_call_metadata"]["name"]: float(str(row["spend"])) for row in rows
            }
            assert by_tool == {"add": 0.0, "multiply": pytest.approx(0.2)}, rows


def test_mcp_tool_cost_is_added_to_the_team_spend_and_stamped_on_the_row(gateway: Gateway) -> None:
    """A key in a fresh team calls a server priced at `default_cost_per_query: 0.25`; the team grants the server
    and the key inherits it, since /key/generate refuses a team key whose servers are outside the team's grant
    (litellm/proxy/management_helpers/object_permission_utils.py:755-776). The row must carry the team id and `LiteLLM_TeamTable.spend` for that team must reach 0.25, the same way an LLM call charges
    the team. A port that only charged the key would let team budgets leak through MCP."""
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "dec1team" + uuid.uuid4().hex[:8]
        identity: Final = _priced(scenario, peer, alias, {"default_cost_per_query": 0.25})
        team: Final = scenario.team(object_permission={"mcp_servers": [identity]})
        key: Final = scenario.key(team_id=team)
        assert McpCaller(gateway, key, "mcp").call(f"{alias}-add", {"a": 2, "b": 3}).text == "5"
        rows: Final = _of_type(spend_rows(key, 1), "call_mcp_tool")
        assert rows[0]["team_id"] == team, rows
        spent: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team,)),
            lambda found: bool(found) and float(str(found[0]["spend"])) >= 0.25,
            seconds=70,
        )
        assert float(str(spent[0]["spend"])) == pytest.approx(0.25), spent


def test_mcp_tool_cost_is_added_to_the_owning_user_spend(gateway: Gateway) -> None:
    """A key owned by a fresh internal user calls a server priced at 0.4. The row's user must be that
    user id and `LiteLLM_UserTable.spend` must reach 0.4."""
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "dec1user" + uuid.uuid4().hex[:8]
        identity: Final = _priced(scenario, peer, alias, {"default_cost_per_query": 0.4})
        user: Final = scenario.user()
        key: Final = scenario.key(user_id=user, object_permission={"mcp_servers": [identity]})
        assert McpCaller(gateway, key, "mcp").call(f"{alias}-add", {"a": 2, "b": 3}).text == "5"
        rows: Final = _of_type(spend_rows(key, 1), "call_mcp_tool")
        assert rows[0]["user_id"] == user, rows
        spent: Final = eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_UserTable" WHERE user_id = %s', (user,)),
            lambda found: bool(found) and float(str(found[0]["spend"])) >= 0.4,
            seconds=70,
        )
        assert float(str(spent[0]["spend"])) == pytest.approx(0.4), spent


def test_a_tool_that_returns_an_error_is_still_logged_with_its_server_and_tool(gateway: Gateway) -> None:
    """The peer's `fail` tool raises, so the gateway returns `isError: true`. A row for `fail` must still
    land with `call_type = call_mcp_tool` and `mcp_tool_call_metadata.mcp_server_name == <alias>`.
    UNVERIFIED: whether the row's status is `success` (the tool call completed at the protocol level) or
    `failure`; the test accepts either and pins only that exactly one row exists for the call."""
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "dec1err" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        outcome: Final = McpCaller(gateway, key, "mcp").call(f"{alias}-fail", {})
        assert outcome.error is not None, outcome.raw
        rows: Final = _of_type(spend_rows(key, 1), "call_mcp_tool")
        assert len(rows) == 1, rows
        assert rows[0]["status"] in ("success", "failure"), rows
        assert metadata_of(rows[0])["mcp_tool_call_metadata"]["mcp_server_name"] == alias, rows


def test_a_protocol_tools_list_writes_one_zero_cost_list_row(gateway: Gateway) -> None:
    """A granted key sends one `tools/list` to `/mcp`. `list_gateway_tools` logs it, so exactly one row
    with `call_type = list_mcp_tools`, `model = "MCP: list_tools"`, spend 0 and status success must land.
    UNVERIFIED: `metadata.spend_logs_metadata` is assumed to carry `mcp_operation: list_tools` and
    `source: mcp_protocol` as set in `_get_tools_from_mcp_servers`."""
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "dec1list" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        assert f"{alias}-add" in McpCaller(gateway, key, "mcp").list_tools().tools
        rows: Final = _of_type(spend_rows(key, 1), "list_mcp_tools")
        assert len(rows) == 1, rows
        assert rows[0]["model"] == "MCP: list_tools", rows
        assert float(str(rows[0]["spend"])) == 0.0, rows
        assert rows[0]["status"] == "success", rows
        logged: Final = metadata_of(rows[0]).get("spend_logs_metadata")
        assert isinstance(logged, dict), rows
        assert logged.get("mcp_operation") == "list_tools" and logged.get("source") == "mcp_protocol", logged


def test_a_rest_tools_list_writes_no_list_row(gateway: Gateway) -> None:
    """The same key lists through `GET /mcp-rest/tools/list` (whose default is
    `log_list_tools_to_spendlogs=False`), then makes one REST tool call as a flush marker. Once the call's
    row has landed, there must be zero `list_mcp_tools` rows for the key."""
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "dec1rlist" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(gateway, key, "rest")
        assert caller.list_tools().ok
        assert caller.call(f"{alias}-add", {"a": 2, "b": 3}, identity).text == "5"
        rows: Final = spend_rows(key, 1)
        assert _of_type(rows, "call_mcp_tool"), rows
        assert _of_type(rows, "list_mcp_tools") == [], rows


def test_an_unknown_jsonrpc_method_is_refused_and_logged_as_a_rejected_request(gateway: Gateway) -> None:
    """A granted key sends method `dec1/unknown` with id 7 to `/mcp`. The SDK answers JSON-RPC error code
    -32601, and `log_mcp_sdk_rejection` writes one failure row with `model = "MCP: request_rejected"`
    whose `metadata.spend_logs_metadata.mcp_sdk_rejection` has `method = dec1/unknown`, `request_id = 7`,
    `error_code = -32601`, `error_message = Method not found` and `stage = sdk_dispatch`. The peer is
    never contacted."""
    with mcp_peer() as peer, gateway.scenario() as scenario:
        identity: Final = register_mcp(scenario, peer, "dec1rej" + uuid.uuid4().hex[:8])
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        peer.drain()
        error: Final = rpc_error(rpc(gateway, "/mcp", "dec1/unknown", {}, key=key, identity=7))
        assert error["code"] == -32601, error
        rows: Final = [row for row in spend_rows(key, 1) if row["model"] == "MCP: request_rejected"]
        assert len(rows) == 1, rows
        assert rows[0]["status"] == "failure", rows
        rejection: Final = metadata_of(rows[0])["spend_logs_metadata"]["mcp_sdk_rejection"]
        assert rejection["method"] == "dec1/unknown", rejection
        assert rejection["request_id"] == "7", rejection
        assert rejection["error_code"] == -32601, rejection
        assert rejection["error_message"] == "Method not found", rejection
        assert rejection["stage"] == "sdk_dispatch", rejection
        assert tool_calls(peer.drain()) == ()


def test_a_tools_call_with_invalid_params_is_logged_as_invalid_request_parameters(gateway: Gateway) -> None:
    """`tools/call` with `params: {"arguments": {}}` (no `name`) fails SDK validation: the reply is
    JSON-RPC -32602 and one `MCP: request_rejected` row lands with `error_code = -32602` and
    `error_message = Invalid request parameters`."""
    with mcp_peer() as peer, gateway.scenario() as scenario:
        identity: Final = register_mcp(scenario, peer, "dec1inv" + uuid.uuid4().hex[:8])
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        error: Final = rpc_error(rpc(gateway, "/mcp", "tools/call", {"arguments": {}}, key=key))
        assert error["code"] == -32602, error
        rows: Final = [row for row in spend_rows(key, 1) if row["model"] == "MCP: request_rejected"]
        assert len(rows) == 1, rows
        rejection: Final = metadata_of(rows[0])["spend_logs_metadata"]["mcp_sdk_rejection"]
        assert rejection["error_code"] == -32602 and rejection["error_message"] == "Invalid request parameters", (
            rejection
        )


def test_a_notification_without_an_id_is_never_logged_as_a_rejection(gateway: Gateway) -> None:
    """An unknown notification (`dec1/notify`, no `id`) is accepted with HTTP 202 or 200 and
    `_sdk_rejection_code` returns None for requests without an id, so after a flush-marker tool call no
    `MCP: request_rejected` row may exist for the key."""
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "dec1note" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        notified: Final = gateway.client.post(
            "/mcp",
            json={"jsonrpc": "2.0", "method": "dec1/notify", "params": {}},
            headers={"x-litellm-api-key": key, "Accept": "application/json, text/event-stream"},
        )
        assert notified.status_code in (200, 202), notified.text
        assert McpCaller(gateway, key, "mcp").call(f"{alias}-add", {"a": 1, "b": 1}).text == "2"
        rows: Final = spend_rows(key, 1)
        assert _of_type(rows, "call_mcp_tool"), rows
        assert [row for row in rows if row["model"] == "MCP: request_rejected"] == [], rows
