"""Tool name prefixes, server naming rules and their collisions on the Python MCP gateway.

The contract: every tool a server exposes is published as `<prefix><separator><tool>`, where the
separator is `MCP_TOOL_PREFIX_SEPARATOR` (default `-`) and the prefix is the server's alias, else its
server name, else its server id (`get_server_prefix` in `litellm/proxy/mcp_server/utils.py`). Spaces in
an alias or server name become underscores. Because the prefix is cut at the separator, a server name or
alias containing the separator is refused at registration with HTTP 400. With
`LITELLM_USE_SHORT_MCP_TOOL_PREFIX` on, the prefix is a deterministic three character id derived from the
server id, and the old alias spelling still routes (`iter_known_server_prefixes`).

A Rust MCP host that rebuilds this naming (it uses `alias or server_name` with spaces replaced and a
hard coded `-`, `gateway-mcp/src/native.rs`) breaks every saved agent configuration if any rule drifts,
so each rule is pinned here.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path
from typing import Final

import pytest
from integration._support.client import Gateway
from integration._support.mcp import McpCaller, listed_tools, mcp_peer, register_mcp, tool_calls
from integration._support.process import owned_proxy

from tests.integration.mcp.dec1_mcp_support import config_with_servers, first_text, rpc, rpc_result, server_ids_by_name

pytestmark: Final = pytest.mark.shares_worker_state

_BASE62: Final = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_BASE52: Final = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _short_prefix(server_id: str) -> str:
    """Re-implementation of `compute_short_server_prefix(server_id, 0)` from `utils.py`, written from the
    product's documented algorithm so the test does not call the product to get its expected value."""
    value = int.from_bytes(hashlib.sha256(server_id.encode()).digest()[:8], "big")
    chars: list[str] = []
    for position in range(3):
        alphabet = _BASE52 if position == 2 else _BASE62
        value, index = divmod(value, len(alphabet))
        chars.append(alphabet[index])
    return "".join(reversed(chars))


def _unique(stem: str) -> str:
    return stem + uuid.uuid4().hex[:8]


def test_an_alias_with_spaces_is_published_with_underscores_and_routes_by_that_spelling(gateway: Gateway) -> None:
    """A server is registered with alias `dec1 space <hex>`. `validate_and_normalize_mcp_server_payload`
    replaces spaces with underscores, so `tools/list` on `/mcp` must name its tool `dec1_space_<hex>-add`
    and a `tools/call` by that name must return `5` for `{a: 2, b: 3}`. A host that kept the space or
    used another replacement would publish names some model APIs reject."""
    with mcp_peer() as peer, gateway.scenario() as scenario:
        suffix: Final = uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, f"dec1 space {suffix}")
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        listed: Final = McpCaller(gateway, key, "mcp").list_tools()
        assert f"dec1_space_{suffix}-add" in listed.tools, listed.raw
        assert not any(" " in name for name in listed.tools), listed.raw
        called: Final = McpCaller(gateway, key, "mcp").call(f"dec1_space_{suffix}-add", {"a": 2, "b": 3})
        assert called.text == "5", called.raw


def test_a_server_name_containing_the_separator_is_refused_with_the_exact_message(gateway: Gateway) -> None:
    """`POST /v1/mcp/server` with `server_name: dec1-bad-<hex>` must answer HTTP 400 with
    `detail.error` equal to `Server name cannot contain '-'. Use an alternative character instead Found:
    dec1-bad-<hex>` (`validate_mcp_server_name`), and no row may be written. A hyphen in a prefix would
    make the first-separator split route calls to the wrong server."""
    name: Final = f"dec1-bad-{uuid.uuid4().hex[:8]}"
    with mcp_peer() as peer:
        response: Final = gateway.request("POST", "/v1/mcp/server", {"server_name": name, **peer.registration()})
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == {
        "error": f"Server name cannot contain '-'. Use an alternative character instead Found: {name}"
    }, response.text
    assert gateway.rows('SELECT server_id FROM "LiteLLM_MCPServerTable" WHERE server_name = %s', (name,)) == []


def test_an_alias_containing_the_separator_is_refused_even_when_the_server_name_is_valid(gateway: Gateway) -> None:
    """`server_name` is a valid `dec1ok<hex>` but `alias` is `dec1-alias-<hex>`. The alias goes through the
    same check, so the answer must be HTTP 400 naming the alias in `detail.error`
    (`Found: dec1-alias-<hex>`)."""
    name: Final = _unique("dec1ok")
    alias: Final = f"dec1-alias-{uuid.uuid4().hex[:8]}"
    with mcp_peer() as peer:
        response: Final = gateway.request(
            "POST", "/v1/mcp/server", {"server_name": name, "alias": alias, **peer.registration()}
        )
    assert response.status_code == 400, response.text
    assert alias in str(response.json()["detail"]["error"]), response.text


def test_separator_validation_runs_before_the_admin_check(gateway: Gateway) -> None:
    """A non-admin key posts a server whose name contains `-`. `add_mcp_server` validates the payload
    before the PROXY_ADMIN check, so the caller gets HTTP 400 (the naming error), not the 403
    `User does not have permission to create mcp servers...`. Pinned because a rewrite that authorizes
    first changes which error an integrator sees."""
    with mcp_peer() as peer, gateway.scenario() as scenario:
        key: Final = scenario.key()
        name: Final = f"dec1-order-{uuid.uuid4().hex[:8]}"
        response: Final = gateway.request(
            "POST", "/v1/mcp/server", {"server_name": name, **peer.registration()}, key=key
        )
    assert response.status_code == 400, response.text
    assert "Server name cannot contain '-'" in response.text, response.text


@pytest.mark.parametrize("reserved", ["all-team-mcpservers", "all-proxy-mcpservers"])
def test_reserved_special_server_ids_are_refused_at_registration(gateway: Gateway, reserved: str) -> None:
    """Posting a server with `server_id` set to one of the `SpecialMCPServerName` values must answer HTTP
    400 with `detail.error` `MCP Server with id <id> is special and cannot be used.`, because those ids are
    grant wildcards on keys and teams."""
    with mcp_peer() as peer:
        response: Final = gateway.request(
            "POST",
            "/v1/mcp/server",
            {"server_name": _unique("dec1reserved"), "server_id": reserved, **peer.registration()},
        )
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == {"error": f"MCP Server with id {reserved} is special and cannot be used."}


def test_a_server_registered_without_an_alias_is_prefixed_by_its_normalized_server_name(gateway: Gateway) -> None:
    """A server is created with only `server_name: dec1 noalias <hex>` (no alias). The alias defaults to
    the normalized server name, so the detail endpoint must return `alias == dec1_noalias_<hex>` and the
    REST listing for the key must name the tool `dec1_noalias_<hex>-add`."""
    suffix: Final = uuid.uuid4().hex[:8]
    with mcp_peer() as peer, gateway.scenario() as scenario:
        response: Final = gateway.request(
            "POST", "/v1/mcp/server", {"server_name": f"dec1 noalias {suffix}", **peer.registration()}
        )
        assert response.status_code == 201, response.text
        identity: Final = str(response.json()["server_id"])
        scenario.cleanups.callback(gateway.request, "DELETE", f"/v1/mcp/server/{identity}")
        assert response.json()["alias"] == f"dec1_noalias_{suffix}", response.text
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        listed: Final = McpCaller(gateway, key, "mcp").list_tools()
        assert f"dec1_noalias_{suffix}-add" in listed.tools, listed.raw


def test_two_servers_whose_aliases_differ_only_by_a_space_and_underscore_collide(gateway: Gateway) -> None:
    """Alias `dec1 col <hex>` normalizes to `dec1_col_<hex>`, the same as a second server's alias
    `dec1_col_<hex>`. The second registration must be refused (HTTP 400 or 409), so two servers can never
    publish the same prefix. UNVERIFIED: the duplicate check in `create_mcp_server_if_identifier_free`
    is assumed to compare normalized aliases; `test_duplicate_alias_is_rejected_so_tool_prefixes_cannot_collide`
    covers only identical spellings."""
    suffix: Final = uuid.uuid4().hex[:8]
    with mcp_peer() as first, mcp_peer() as second, gateway.scenario() as scenario:
        register_mcp(scenario, first, f"dec1 col {suffix}")
        clash: Final = gateway.request(
            "POST",
            "/v1/mcp/server",
            {"server_name": f"dec1col2{suffix}", "alias": f"dec1_col_{suffix}", **second.registration()},
        )
        if clash.status_code == 201:
            gateway.request("DELETE", f"/v1/mcp/server/{clash.json()['server_id']}")
        assert clash.status_code in (400, 409), clash.text


def test_rest_listing_of_one_server_returns_bare_names_with_server_identity_in_mcp_info(gateway: Gateway) -> None:
    """`GET /mcp-rest/tools/list?server_id=<id>` lists one server through `_list_server_tools` with
    `add_prefix=False`. UNVERIFIED: each entry is assumed to carry the bare or prefixed name together with
    `mcp_info.server_id == <id>`; the test pins that every listed tool carries that server id and that
    `add` is reachable by name suffix. The Rust host's REST list returns bare names plus
    `mcp_info {server_id, server_name, alias}` (`gateway-mcp/src/lib.rs` `RestTool`)."""
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = _unique("dec1rest")
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        tools: Final = listed_tools(gateway, key, identity)
        assert any(name.endswith("add") for name in tools), tools
        for tool in tools.values():
            info = tool.get("mcp_info")
            assert isinstance(info, dict) and info.get("server_id") == identity, tool


def test_short_prefix_mode_publishes_a_three_character_prefix_derived_from_the_server_id(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A proxy boots with `LITELLM_USE_SHORT_MCP_TOOL_PREFIX=true` and one config-declared server. Its
    tools must be listed as `<p>-add` where `<p>` is the three character base62 id of the server id (first
    character alphabetic), and calling `<p>-add` must return `5`. A host that ignores the flag would
    republish every tool under a new name."""
    with mcp_peer() as peer:
        name: Final = _unique("dec1short")
        config: Final = config_with_servers(tmp_path, {name: {**peer.registration(), "allow_all_keys": True}})
        with owned_proxy(gateway, tmp_path, {"LITELLM_USE_SHORT_MCP_TOOL_PREFIX": "true"}, config=config) as owned:
            identity: Final = server_ids_by_name(owned)[name]
            prefix: Final = _short_prefix(identity)
            assert prefix[0].isalpha(), prefix
            listed: Final = McpCaller(owned, owned.key, "mcp").list_tools()
            assert f"{prefix}-add" in listed.tools, (prefix, listed.raw)
            assert f"{name}-add" not in listed.tools, listed.raw
            called: Final = McpCaller(owned, owned.key, "mcp").call(f"{prefix}-add", {"a": 2, "b": 3})
            assert called.text == "5", called.raw


def test_short_prefix_mode_still_routes_the_alias_spelling_a_client_saved_earlier(
    gateway: Gateway, tmp_path: Path
) -> None:
    """Same proxy as above. A client that saved `<name>-multiply` before the flag was turned on calls it
    by that name: `iter_known_server_prefixes` still accepts the alias and server name, so the call must
    return `6` for `{a: 2, b: 3}` and reach the peer exactly once."""
    with mcp_peer() as peer:
        name: Final = _unique("dec1legacy")
        config: Final = config_with_servers(tmp_path, {name: {**peer.registration(), "allow_all_keys": True}})
        with owned_proxy(gateway, tmp_path, {"LITELLM_USE_SHORT_MCP_TOOL_PREFIX": "true"}, config=config) as owned:
            peer.drain()
            called: Final = McpCaller(owned, owned.key, "mcp").call(f"{name}-multiply", {"a": 2, "b": 3})
            assert called.text == "6", called.raw
            assert len(tool_calls(peer.drain())) == 1


def test_a_custom_separator_changes_published_names_and_allows_hyphens_in_names(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A proxy boots with `MCP_TOOL_PREFIX_SEPARATOR=__` and a config-declared server named
    `dec1-sep-<hex>` (a hyphen is legal once the separator is `__`). Tools must be listed as
    `dec1-sep-<hex>__add` and a call by that name must return `5`. The Rust host hard codes `-`
    (`native.rs`), so this is a known gap for any deployment that set the variable."""
    with mcp_peer() as peer:
        name: Final = f"dec1-sep-{uuid.uuid4().hex[:8]}"
        config: Final = config_with_servers(tmp_path, {name: {**peer.registration(), "allow_all_keys": True}})
        with owned_proxy(gateway, tmp_path, {"MCP_TOOL_PREFIX_SEPARATOR": "__"}, config=config) as owned:
            listed: Final = McpCaller(owned, owned.key, "mcp").list_tools()
            assert f"{name}__add" in listed.tools, listed.raw
            called: Final = McpCaller(owned, owned.key, "mcp").call(f"{name}__add", {"a": 2, "b": 3})
            assert called.text == "5", called.raw


def test_a_custom_separator_is_refused_inside_a_registered_server_name(gateway: Gateway, tmp_path: Path) -> None:
    """With `MCP_TOOL_PREFIX_SEPARATOR=__`, `POST /v1/mcp/server` with `server_name: dec1__x<hex>` must be
    HTTP 400 with `Server name cannot contain '__'` in the body, while a name with `-` is now accepted
    with 201."""
    with mcp_peer() as peer, owned_proxy(gateway, tmp_path, {"MCP_TOOL_PREFIX_SEPARATOR": "__"}) as owned:
        refused: Final = owned.request(
            "POST", "/v1/mcp/server", {"server_name": f"dec1__x{uuid.uuid4().hex[:6]}", **peer.registration()}
        )
        assert refused.status_code == 400, refused.text
        assert "Server name cannot contain '__'" in refused.text, refused.text
        accepted: Final = owned.request(
            "POST", "/v1/mcp/server", {"server_name": f"dec1-ok-{uuid.uuid4().hex[:6]}", **peer.registration()}
        )
        assert accepted.status_code == 201, accepted.text
        owned.request("DELETE", f"/v1/mcp/server/{accepted.json()['server_id']}")


def test_prefixed_call_to_a_tool_the_server_does_not_have_never_reaches_another_server(gateway: Gateway) -> None:
    """Two servers `dec1a<hex>` and `dec1b<hex>` are granted to one key. A `tools/call` for
    `dec1a<hex>-nosuch` must fail (JSON-RPC error or `isError: true`) and neither peer may receive a
    `tools/call`, so an unknown suffix is never retried on a sibling server whose name happens to match."""
    with mcp_peer() as first, mcp_peer() as second, gateway.scenario() as scenario:
        alias_a: Final = _unique("dec1a")
        a: Final = register_mcp(scenario, first, alias_a)
        b: Final = register_mcp(scenario, second, _unique("dec1b"))
        key: Final = scenario.key(object_permission={"mcp_servers": [a, b]})
        first.drain()
        second.drain()
        response: Final = rpc(gateway, "/mcp", "tools/call", {"name": f"{alias_a}-nosuch", "arguments": {}}, key=key)
        body_text: Final = response.text
        assert response.status_code != 200 or '"error"' in body_text or '"isError":true' in body_text.replace(" ", ""), body_text
        assert [call for call in tool_calls(first.drain()) if call["body"]["params"]["name"] != "nosuch"] == []
        assert tool_calls(second.drain()) == ()


def test_server_scoped_path_accepts_the_bare_tool_name(gateway: Gateway) -> None:
    """Through `/<alias>/mcp` a single server is in scope. A `tools/call` for the bare name `add` must
    still resolve to that server and return `5`. The Rust host does the same through `resolve_name`'s
    `scoped && servers.len() == 1` arm for prompts and by matching bare names for tools."""
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = _unique("dec1bare")
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        result: Final = rpc_result(
            rpc(gateway, f"/{alias}/mcp", "tools/call", {"name": "add", "arguments": {"a": 2, "b": 3}}, key=key)
        )
        assert first_text(result) == "5", result
