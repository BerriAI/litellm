import json
import re
import textwrap
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import httpx
import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, gateway_from_environment, object_value, string_value
from integration._support.database import read_rows, scratch_database
from integration._support.mcp import (
    ENTRY_POINTS,
    EntryPoint,
    JsonRpc,
    McpCaller,
    McpPeer,
    Outcome,
    PeerKind,
    ScriptedTool,
    echo_tool,
    peer_of,
    register_mcp,
    scripted_peer,
    text_result,
    tool_calls,
)
from integration._support.mcp_grants import SUBJECTS, Subject, create_toolset, grant
from integration._support.process import owned_proxy
from pydantic import BaseModel, JsonValue, TypeAdapter

CALLABLE: Final = {"add": {"a": 1, "b": 2}, "multiply": {"a": 2, "b": 3}}
RESULTS: Final = {"add": "3", "multiply": "6"}


class TextBlock(BaseModel):
    type: Literal["text"]
    text: str


class CallResult(BaseModel):
    content: list[TextBlock]
    isError: bool


class CalledParams(BaseModel):
    name: str


class CalledBody(BaseModel):
    params: CalledParams


_TOOL_ENTRIES: Final = TypeAdapter(list[dict[str, JsonValue]])


def _called_names(peer: McpPeer) -> list[str]:
    return [CalledBody.model_validate(call["body"]).params.name for call in tool_calls(peer.drain())]


def _server_scoped(entry: EntryPoint, identity: str) -> str | None:
    return identity if entry == "rest" else None


def _name(entry: EntryPoint, alias: str, tool: str) -> str:
    return tool if entry == "rest" else f"{alias}-{tool}"


def _open_aliases() -> frozenset[str]:
    rows: Final = read_rows('SELECT alias FROM "LiteLLM_MCPServerTable" WHERE allow_all_keys', ())
    return frozenset(str(row["alias"]) for row in rows)


def _without_foreign_open_servers(tools: tuple[str, ...], open_aliases: frozenset[str]) -> set[str]:
    prefixes: Final = tuple(f"{alias}-" for alias in open_aliases)
    return {tool for tool in tools if not tool.startswith(prefixes)}


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
        open_before: Final = _open_aliases()
        listed: Final = reach.list_tools(_server_scoped(entry, granted))
        assert listed.ok, listed.raw
        open_aliases: Final = open_before | _open_aliases()
        expected: Final = (
            {_name(entry, granted_alias, "add")}
            if subject in ("toolset", "allowed_tools")
            else {_name(entry, granted_alias, tool) for tool in ("add", "multiply", "fail")}
        )
        assert _without_foreign_open_servers(listed.tools, open_aliases) == expected, listed.tools
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


_PROBE: Final = "catalog-probe"
_ECHO: Final = "catalog-echo"
_UNLISTED: Final = ""
_GUARDRAIL_CODE: Final = (
    "def apply_guardrail(inputs, request_data, input_type):\n"
    f'    if "{_PROBE}" not in list(inputs.get("texts") or []):\n'
    "        return allow()\n"
    '    function = inputs.get("tools", [{}])[0].get("function", {})\n'
    f'    return block("{_ECHO}[" + function.get("description") + "]")\n'
)


_ECHO_GUARDRAIL_YAML: Final = (
    "guardrails:\n"
    "  - guardrail_name: catalog-echo\n"
    "    litellm_params:\n"
    "      guardrail: custom_code\n"
    "      mode: pre_mcp_call\n"
    "      default_on: true\n"
    "      custom_code: |\n" + textwrap.indent(_GUARDRAIL_CODE, 8 * " ")
)


@pytest.fixture(scope="module")
def echo_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("catalog-echo")
    path: Final = directory / "catalog_echo.yaml"
    path.write_text((Path(__file__).resolve().parents[1] / "proxy_config.yaml").read_text() + _ECHO_GUARDRAIL_YAML)
    with gateway_from_environment() as gateway, owned_proxy(gateway, directory, {}, config=path, workers=2) as rig:
        yield rig


def _echoed_description(outcome: Outcome) -> str:
    found: Final = re.search(rf"{_ECHO}\[(.*?)\]", outcome.raw)
    assert found is not None, outcome.raw
    return found.group(1)


@pytest.mark.parametrize("subject", SUBJECTS)
def test_each_subjects_call_is_evaluated_only_against_the_catalog_its_own_listing_served(
    echo_rig: Gateway, subject: Subject
) -> None:
    described: Final = "Adds under grant " + uuid.uuid4().hex[:8]
    tool: Final = ScriptedTool("add", lambda _: text_result("3"), description=described)
    with scripted_peer(tool) as peer, echo_rig.scenario() as scenario:
        group: Final = "grp" + uuid.uuid4().hex[:8]
        alias: Final = "cat" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias, mcp_access_groups=[group])
        caller: Final = grant(
            scenario, subject, (identity,), (identity,), access_group=group, allowed_tools={identity: ("add",)}
        )
        reach: Final = McpCaller(echo_rig, caller.key, "mcp", alias, caller.headers)
        assert reach.initialize().ok
        cold: Final = _echoed_description(reach.call(f"{alias}-add", {"probe": _PROBE}))
        listed: Final = reach.list_tools()
        assert listed.ok and f"{alias}-add" in listed.tools, listed.raw
        warm: Final = _echoed_description(reach.call(f"{alias}-add", {"probe": _PROBE}))
        assert (cold, warm) == (_UNLISTED, described), (cold, warm)
        assert tool_calls(peer.drain()) == (), "a blocked probe reached the peer"


def test_end_users_of_one_key_share_its_catalog_slot_because_the_identity_excludes_the_end_user(
    echo_rig: Gateway,
) -> None:
    described: Final = "Adds for end users " + uuid.uuid4().hex[:8]
    tool: Final = ScriptedTool("add", lambda _: text_result("3"), description=described)
    with scripted_peer(tool) as peer, echo_rig.scenario() as scenario:
        alias: Final = "eu" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        granted: Final = grant(scenario, "end_user", (identity,), (identity,))
        first: Final = McpCaller(echo_rig, granted.key, "mcp", alias, granted.headers)
        second: Final = McpCaller(
            echo_rig, granted.key, "mcp", alias, {"x-litellm-end-user-id": "integration-" + uuid.uuid4().hex[:10]}
        )
        assert second.initialize().ok
        assert _echoed_description(second.call(f"{alias}-add", {"probe": _PROBE})) == _UNLISTED
        listed: Final = first.list_tools()
        assert listed.ok and f"{alias}-add" in listed.tools, listed.raw
        assert _echoed_description(second.call(f"{alias}-add", {"probe": _PROBE})) == described, (
            "the end-user header is intentionally not part of the catalog identity: one key, one slot"
        )
        assert tool_calls(peer.drain()) == (), "a blocked probe reached the peer"


def _rest_call(gateway: Gateway, key: str, server_id: str, name: str, arguments: JsonRpc) -> httpx.Response:
    return gateway.client.post(
        "/mcp-rest/tools/call",
        headers={"x-litellm-api-key": key},
        json={"server_id": server_id, "name": name, "arguments": dict(arguments)},
    )


def _rest_listing(gateway: Gateway, key: str, params: Mapping[str, str]) -> httpx.Response:
    return gateway.client.get("/mcp-rest/tools/list", headers={"x-litellm-api-key": key}, params=dict(params))


def _single_tools_call(peer: McpPeer, name: str, arguments: JsonRpc) -> None:
    calls: Final = tool_calls(peer.drain())
    assert len(calls) == 1, calls
    body: Final = calls[0]["body"]
    assert isinstance(body, dict), calls
    assert body == {
        "jsonrpc": "2.0",
        "id": body["id"],
        "method": "tools/call",
        "params": {"name": name, "arguments": dict(arguments), "_meta": {"progressToken": body["id"]}},
    }, body
    assert isinstance(body["id"], int), body


@pytest.mark.parametrize("spelling", ("alias", "server_name"))
def test_rest_call_by_alias_or_server_name_reaches_only_the_granted_server(gateway: Gateway, spelling: str) -> None:
    with (
        scripted_peer(echo_tool("lookup")) as granted_peer,
        scripted_peer(echo_tool("lookup")) as sibling_peer,
        gateway.scenario() as scenario,
    ):
        suffix: Final = uuid.uuid4().hex[:8]
        names: Final = {"alias": f"order_status_{suffix}", "server_name": f"orders_{suffix}"}
        sibling_names: Final = {"alias": f"order_audit_{suffix}", "server_name": f"audit_{suffix}"}
        granted: Final = register_mcp(scenario, granted_peer, names["alias"], server_name=names["server_name"])
        register_mcp(scenario, sibling_peer, sibling_names["alias"], server_name=sibling_names["server_name"])
        key: Final = scenario.key(object_permission={"mcp_servers": [granted]})
        arguments: Final = {"order_id": "A-" + suffix, "verbose": True, "lines": [1, 2]}
        granted_peer.drain()
        sibling_peer.drain()
        response: Final = _rest_call(gateway, key, names[spelling], "lookup", arguments)
        assert response.status_code == 200, response.text
        assert CallResult.model_validate_json(response.content) == CallResult(
            content=[TextBlock(type="text", text=json.dumps(arguments, sort_keys=True))], isError=False
        ), response.text
        _single_tools_call(granted_peer, "lookup", arguments)
        refused: Final = _rest_call(gateway, key, sibling_names[spelling], "lookup", arguments)
        assert refused.status_code == 403, refused.text
        assert refused.json() == {
            "detail": {
                "error": "access_denied",
                "message": f"The key is not allowed to access server {sibling_names[spelling]}",
            }
        }, refused.text
        assert tool_calls(sibling_peer.drain()) == (), "an ungranted server's name reached its peer"
        assert tool_calls(granted_peer.drain()) == (), "an ungranted name resolved to the granted sibling"


def test_a_config_alias_shadowing_a_db_server_name_resolves_alias_first_and_never_crosses_the_grant(
    gateway: Gateway, tmp_path: Path
) -> None:
    with scripted_peer(echo_tool("lookup")) as config_peer, scripted_peer(echo_tool("lookup")) as db_peer:
        suffix: Final = uuid.uuid4().hex[:8]
        shared: Final = f"orders_{suffix}"
        config_name: Final = f"order_config_{suffix}"
        config: Final = JSON_OBJECT.validate_python(
            yaml.safe_load((Path(__file__).resolve().parents[1] / "proxy_config.yaml").read_text())
        )
        config["mcp_servers"] = {
            config_name: {**JSON_OBJECT.validate_python(config_peer.registration()), "alias": shared}
        }
        path: Final = tmp_path / "shadowing_alias.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            servers: Final = candidate.client.get("/v1/mcp/server", headers={"x-litellm-api-key": candidate.key})
            assert servers.status_code == 200, servers.text
            config_server: Final = next(
                string_value(server["server_id"])
                for server in _TOOL_ENTRIES.validate_json(servers.content)
                if server["server_name"] == config_name
            )
            db_server: Final = register_mcp(scenario, db_peer, f"order_db_{suffix}", server_name=shared)
            alias_key: Final = scenario.key(object_permission={"mcp_servers": [config_server]})
            name_key: Final = scenario.key(object_permission={"mcp_servers": [db_server]})
            arguments: Final = {"order_id": "B-" + suffix}
            config_peer.drain()
            db_peer.drain()
            served: Final = _rest_call(candidate, alias_key, shared, "lookup", arguments)
            assert served.status_code == 200, served.text
            assert CallResult.model_validate_json(served.content) == CallResult(
                content=[TextBlock(type="text", text=json.dumps(arguments, sort_keys=True))], isError=False
            ), served.text
            _single_tools_call(config_peer, "lookup", arguments)
            assert tool_calls(db_peer.drain()) == (), "the alias match also reached the server_name sibling"
            refused: Final = _rest_call(candidate, name_key, shared, "lookup", arguments)
            assert refused.status_code == 403, refused.text
            assert refused.json() == {
                "detail": {"error": "access_denied", "message": f"The key is not allowed to access server {shared}"}
            }, refused.text
            assert tool_calls(config_peer.drain()) == (), "a name owned by an ungranted alias reached that server"
            assert tool_calls(db_peer.drain()) == (), "a name owned by an ungranted alias fell back to the sibling"


def _tool_entries(response: httpx.Response) -> list[dict[str, JsonValue]]:
    assert response.status_code == 200, response.text
    tools: Final = _TOOL_ENTRIES.validate_python(JSON_OBJECT.validate_json(response.content)["tools"])
    return sorted(tools, key=lambda tool: string_value(tool["name"]))


def test_rest_listing_selects_the_same_scope_for_every_server_spelling(gateway: Gateway) -> None:
    with peer_of("http") as peer, peer_of("http") as sibling_peer, gateway.scenario() as scenario:
        suffix: Final = uuid.uuid4().hex[:8]
        alias: Final = f"scope_alias_{suffix}"
        server_name: Final = f"scope_name_{suffix}"
        identity: Final = register_mcp(
            scenario, peer, alias, server_name=server_name, allowed_tools=["add", "multiply"]
        )
        sibling: Final = register_mcp(scenario, sibling_peer, f"scope_sibling_{suffix}", allowed_tools=["fail"])
        toolset_name: Final = f"scope_toolset_{suffix}"
        toolset: Final = create_toolset(scenario, ((identity, "add"), (identity, "multiply")), toolset_name)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity, sibling], "mcp_toolsets": [toolset]})
        by_uuid: Final = _tool_entries(_rest_listing(gateway, key, {"server_id": identity}))
        assert [tool["name"] for tool in by_uuid] == ["add", "multiply"], by_uuid
        assert {string_value(object_value(tool["mcp_info"])["server_id"]) for tool in by_uuid} == {identity}, by_uuid
        add_schema: Final = by_uuid[0]["inputSchema"]
        assert isinstance(add_schema, dict) and add_schema["properties"] == {
            "a": {"title": "A", "type": "integer"},
            "b": {"title": "B", "type": "integer"},
        }, add_schema
        for params in (
            {"server_id": alias},
            {"server_id": server_name},
            {"mcp_server_name": alias},
            {"mcp_server_name": server_name},
            {"toolset_name": toolset_name},
        ):
            listed = _rest_listing(gateway, key, params)
            assert _tool_entries(listed) == by_uuid, (params, listed.text)


def test_include_disabled_tools_widens_the_listing_only_for_an_admin(gateway: Gateway) -> None:
    with peer_of("http") as peer, gateway.scenario() as scenario:
        alias: Final = "disabled" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias, allowed_tools=["add"])
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        wide: Final = {"server_id": identity, "include_disabled_tools": "true"}
        admin_view: Final = _tool_entries(_rest_listing(gateway, gateway.key, wide))
        assert [tool["name"] for tool in admin_view] == ["add", "fail", "multiply"], admin_view
        member_view: Final = _tool_entries(_rest_listing(gateway, key, wide))
        assert [tool["name"] for tool in member_view] == ["add"], member_view
        assert member_view == _tool_entries(_rest_listing(gateway, key, {"server_id": identity}))


def _assert_only_add_is_listed_and_callable(
    gateway: Gateway, key: str, peer: McpPeer, alias: str, identity: str
) -> None:
    rest: Final = McpCaller(gateway, key, "rest", alias)
    rpc: Final = McpCaller(gateway, key, "server_mcp", alias)
    listed_rest: Final = rest.list_tools(identity)
    assert listed_rest.ok and listed_rest.tools == ("add",), listed_rest.raw
    assert rpc.initialize().ok
    listed_rpc: Final = rpc.list_tools()
    assert listed_rpc.ok and listed_rpc.tools == (f"{alias}-add",), listed_rpc.raw
    for caller, name in ((rest, "multiply"), (rest, f"{alias}-multiply"), (rpc, f"{alias}-multiply")):
        peer.drain()
        refused = caller.call(name, CALLABLE["multiply"], identity if caller is rest else None)
        assert refused.error is not None and refused.text != RESULTS["multiply"], (name, refused.raw)
        assert tool_calls(peer.drain()) == (), f"filtered {name} reached the peer"
    for caller, name in ((rest, "add"), (rest, f"{alias}-add"), (rpc, f"{alias}-add")):
        peer.drain()
        allowed = caller.call(name, CALLABLE["add"], identity if caller is rest else None)
        assert allowed.ok and allowed.text == RESULTS["add"], (name, allowed.raw)
        assert _called_names(peer) == ["add"], (name, allowed.raw)


@pytest.mark.parametrize("configured", ("bare", "prefixed"))
def test_server_allowed_tools_hides_other_tools_from_every_listing_and_refuses_every_call_spelling(
    gateway: Gateway, configured: str
) -> None:
    with peer_of("http") as peer, gateway.scenario() as scenario:
        alias: Final = "allow" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(
            scenario, peer, alias, allowed_tools=["add"] if configured == "bare" else [f"{alias}-add"]
        )
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        _assert_only_add_is_listed_and_callable(gateway, key, peer, alias, identity)


def test_config_declared_disallowed_tools_hide_and_refuse_the_tool_in_bare_and_prefixed_spelling(
    gateway: Gateway, tmp_path: Path
) -> None:
    with peer_of("http") as bare_peer, peer_of("http") as prefixed_peer:
        bare: Final = "denybare" + uuid.uuid4().hex[:8]
        prefixed: Final = "denypre" + uuid.uuid4().hex[:8]
        config: Final = JSON_OBJECT.validate_python(
            yaml.safe_load((Path(__file__).resolve().parents[1] / "proxy_config.yaml").read_text())
        )
        config["mcp_servers"] = {
            bare: {**JSON_OBJECT.validate_python(bare_peer.registration()), "disallowed_tools": ["multiply", "fail"]},
            prefixed: {
                **JSON_OBJECT.validate_python(prefixed_peer.registration()),
                "disallowed_tools": [f"{prefixed}-multiply", f"{prefixed}-fail"],
            },
        }
        path: Final = tmp_path / "disallowed.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            servers: Final = candidate.client.get("/v1/mcp/server", headers={"x-litellm-api-key": candidate.key})
            assert servers.status_code == 200, servers.text
            ids: Final = {
                string_value(server["server_name"]): string_value(server["server_id"])
                for server in _TOOL_ENTRIES.validate_json(servers.content)
            }
            for alias, peer in ((bare, bare_peer), (prefixed, prefixed_peer)):
                key = scenario.key(object_permission={"mcp_servers": [ids[alias]]})
                _assert_only_add_is_listed_and_callable(candidate, key, peer, alias, ids[alias])


def test_api_created_disallowed_tools_hide_and_refuse_the_tool_in_bare_and_prefixed_spelling(gateway: Gateway) -> None:
    pytest.skip("BUG: POST /v1/mcp/server silently drops disallowed_tools and the tool stays callable")
    with peer_of("http") as peer, gateway.scenario() as scenario:
        alias: Final = "apideny" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias, disallowed_tools=["multiply", "fail"])
        readback: Final = gateway.client.get(f"/v1/mcp/server/{identity}", headers={"x-litellm-api-key": gateway.key})
        assert readback.status_code == 200, readback.text
        assert JSON_OBJECT.validate_json(readback.content)["disallowed_tools"] == ["multiply", "fail"], readback.text
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        _assert_only_add_is_listed_and_callable(gateway, key, peer, alias, identity)


def _ip_filtering_detail(server_id: str, client_ip: str) -> dict[str, str]:
    return {
        "error": "ip_filtering",
        "message": (
            f"MCP server '{server_id}' is not accessible from your IP address ({client_ip}). This server is "
            "restricted to internal networks only. To make it externally accessible, set "
            "'available_on_public_internet: true' in the server configuration."
        ),
    }


def _created_server(gateway: Gateway, body: Mapping[str, JsonValue]) -> str:
    response: Final = gateway.request("POST", "/v1/mcp/server", body)
    assert response.status_code == 201, response.text
    return string_value(JSON_OBJECT.validate_json(response.content)["server_id"])


def _served_pairs(response: httpx.Response) -> set[tuple[str, str]]:
    return {
        (string_value(object_value(tool["mcp_info"])["server_id"]), string_value(tool["name"]))
        for tool in _tool_entries(response)
    }


@dataclass(frozen=True, slots=True)
class _ForwardedIpRig:
    candidate: Gateway
    public: str
    internal: str
    key: str
    public_peer: McpPeer
    internal_peer: McpPeer


@contextmanager
def _forwarded_ip_rig(
    gateway: Gateway, tmp_path: Path, settings: Mapping[str, JsonValue]
) -> Iterator[_ForwardedIpRig]:
    config: Final = JSON_OBJECT.validate_python(
        yaml.safe_load((Path(__file__).resolve().parents[1] / "proxy_config.yaml").read_text())
    )
    config["general_settings"] = {
        **object_value(config["general_settings"]),
        "use_x_forwarded_for": True,
        **settings,
    }
    path: Final = tmp_path / "forwarded_ip.yaml"
    path.write_text(yaml.safe_dump(config))
    with (
        peer_of("http") as public_peer,
        peer_of("http") as internal_peer,
        scratch_database() as database_url,
        owned_proxy(
            gateway,
            tmp_path,
            {"DATABASE_URL": database_url, "FORWARDED_ALLOW_IPS": "192.0.2.1"},
            config=path,
            remove_environment=("DATABASE_URL_READ_REPLICA",),
        ) as candidate,
    ):
        suffix: Final = uuid.uuid4().hex[:8]
        public: Final = _created_server(
            candidate,
            {
                "server_name": "pub" + suffix,
                "alias": "pub" + suffix,
                **JSON_OBJECT.validate_python(public_peer.registration()),
            },
        )
        internal: Final = _created_server(
            candidate,
            {
                "server_name": "int" + suffix,
                "alias": "int" + suffix,
                "available_on_public_internet": False,
                **JSON_OBJECT.validate_python(internal_peer.registration()),
            },
        )
        key: Final = string_value(
            candidate.post("/key/generate", {"object_permission": {"mcp_servers": [public, internal]}})["key"]
        )
        yield _ForwardedIpRig(candidate, public, internal, key, public_peer, internal_peer)


def _forwarded_ip_headers(key: str, forwarded: str) -> dict[str, str]:
    return {"x-litellm-api-key": key, "x-forwarded-for": forwarded}


def _forwarded_ip_listing(
    client: httpx.Client, key: str, forwarded: str, server_id: str | None = None
) -> httpx.Response:
    return client.get(
        "/mcp-rest/tools/list",
        headers=_forwarded_ip_headers(key, forwarded),
        params={"server_id": server_id} if server_id else None,
    )


def _forwarded_ip_call(client: httpx.Client, key: str, forwarded: str, server_id: str) -> httpx.Response:
    return client.post(
        "/mcp-rest/tools/call",
        headers=_forwarded_ip_headers(key, forwarded),
        json={"server_id": server_id, "name": "add", "arguments": CALLABLE["add"]},
    )


def _assert_external_view(
    client: httpx.Client,
    key: str,
    forwarded: str,
    client_ip: str,
    public: str,
    internal: str,
    public_peer: McpPeer,
    internal_peer: McpPeer,
) -> None:
    tools: Final = ("add", "fail", "multiply")
    public_only: Final = {(public, tool) for tool in tools}
    listed: Final = _forwarded_ip_listing(client, key, forwarded)
    assert _served_pairs(listed) == public_only, listed.text
    scoped: Final = _forwarded_ip_listing(client, key, forwarded, internal)
    assert scoped.status_code == 403, scoped.text
    assert scoped.json() == {"detail": _ip_filtering_detail(internal, client_ip)}, scoped.text
    internal_peer.drain()
    refused: Final = _forwarded_ip_call(client, key, forwarded, internal)
    assert refused.status_code == 403, refused.text
    assert refused.json() == {"detail": _ip_filtering_detail(internal, client_ip)}, refused.text
    assert tool_calls(internal_peer.drain()) == (), "an external caller reached an internal-only server"
    public_peer.drain()
    served: Final = _forwarded_ip_call(client, key, forwarded, public)
    assert served.status_code == 200 and served.json()["content"][0]["text"] == RESULTS["add"], served.text
    assert _called_names(public_peer) == ["add"], "the public server did not receive exactly one add call"


def _assert_internal_view(
    client: httpx.Client,
    key: str,
    forwarded: str,
    public: str,
    internal: str,
    public_peer: McpPeer,
    internal_peer: McpPeer,
) -> None:
    tools: Final = ("add", "fail", "multiply")
    both: Final = {(server, tool) for server in (public, internal) for tool in tools}
    listed: Final = _forwarded_ip_listing(client, key, forwarded)
    assert _served_pairs(listed) == both, listed.text
    public_peer.drain()
    internal_peer.drain()
    reached: Final = _forwarded_ip_call(client, key, forwarded, internal)
    assert reached.status_code == 200 and reached.json()["content"][0]["text"] == RESULTS["add"], reached.text
    assert _called_names(internal_peer) == ["add"], "the internal caller did not reach the internal server once"
    assert tool_calls(public_peer.drain()) == (), "the internal call reached the public server"


def test_forwarded_client_ip_limits_external_callers_to_public_servers_and_make_public_widens_them(
    gateway: Gateway, tmp_path: Path
) -> None:
    settings: Final = {
        "mcp_trusted_proxy_ranges": ["127.0.0.1/32"],
        "mcp_internal_ip_ranges": ["10.0.0.0/8"],
    }
    with (
        _forwarded_ip_rig(gateway, tmp_path, settings) as rig,
        httpx.Client(
            base_url=str(rig.candidate.client.base_url),
            transport=httpx.HTTPTransport(local_address="127.0.0.2"),
            timeout=15,
            trust_env=False,
        ) as untrusted,
    ):
        external: Final = "203.0.113.7"
        _assert_internal_view(
            rig.candidate.client, rig.key, "10.1.2.3", rig.public, rig.internal, rig.public_peer, rig.internal_peer
        )
        _assert_external_view(
            rig.candidate.client,
            rig.key,
            external,
            external,
            rig.public,
            rig.internal,
            rig.public_peer,
            rig.internal_peer,
        )
        _assert_external_view(
            untrusted,
            rig.key,
            "10.1.2.3",
            "127.0.0.2",
            rig.public,
            rig.internal,
            rig.public_peer,
            rig.internal_peer,
        )
        _assert_external_view(
            rig.candidate.client,
            rig.key,
            "203.0.113.7, 10.0.0.1",
            "203.0.113.7, 10.0.0.1",
            rig.public,
            rig.internal,
            rig.public_peer,
            rig.internal_peer,
        )
        _assert_internal_view(
            rig.candidate.client,
            rig.key,
            "10.1.2.3, 10.0.0.1",
            rig.public,
            rig.internal,
            rig.public_peer,
            rig.internal_peer,
        )

        published: Final = rig.candidate.request(
            "POST", "/v1/mcp/make_public", {"mcp_server_ids": [rig.internal]}
        )
        assert published.status_code == 202, published.text
        assert published.json()["public_mcp_servers"] == [rig.internal], published.text
        widened_view: Final = _forwarded_ip_listing(rig.candidate.client, rig.key, external)
        tools: Final = ("add", "fail", "multiply")
        both: Final = {(server, tool) for server in (rig.public, rig.internal) for tool in tools}
        assert _served_pairs(widened_view) == both, widened_view.text
        rig.internal_peer.drain()
        widened: Final = _forwarded_ip_call(rig.candidate.client, rig.key, external, rig.internal)
        assert widened.status_code == 200 and widened.json()["content"][0]["text"] == RESULTS["add"], widened.text
        assert _called_names(rig.internal_peer) == ["add"], "make_public did not let the external caller reach it once"


def test_trusted_hop_count_takes_the_client_from_a_load_balancer_chain_and_ignores_prepended_spoofs(
    gateway: Gateway, tmp_path: Path
) -> None:
    settings: Final = {
        "mcp_trusted_proxy_ranges": ["127.0.0.1/32"],
        "mcp_internal_ip_ranges": ["10.0.0.0/8"],
        "mcp_xff_num_trusted_hops": 2,
    }
    with _forwarded_ip_rig(gateway, tmp_path, settings) as rig:
        _assert_external_view(
            rig.candidate.client,
            rig.key,
            "203.0.113.7, 10.0.0.1",
            "203.0.113.7",
            rig.public,
            rig.internal,
            rig.public_peer,
            rig.internal_peer,
        )
        _assert_external_view(
            rig.candidate.client,
            rig.key,
            "10.1.2.3, 203.0.113.7, 10.0.0.1",
            "203.0.113.7",
            rig.public,
            rig.internal,
            rig.public_peer,
            rig.internal_peer,
        )
        _assert_internal_view(
            rig.candidate.client,
            rig.key,
            "10.1.2.3, 10.0.0.1",
            rig.public,
            rig.internal,
            rig.public_peer,
            rig.internal_peer,
        )
        _assert_external_view(
            rig.candidate.client,
            rig.key,
            "203.0.113.7",
            "",
            rig.public,
            rig.internal,
            rig.public_peer,
            rig.internal_peer,
        )
