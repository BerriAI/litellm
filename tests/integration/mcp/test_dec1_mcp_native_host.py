"""What the Rust MCP host in the native gateway does today: what it refuses at boot and how it serves.

The contract (`litellm/rust/crates/gateway/src/lib.rs` `build_mcp`, `gateway-mcp/src/configured.rs`,
`native.rs`, `routes.rs`, `sessions.rs`, `error.rs`):

At boot, MCP turns on only when `mcp_servers` is non-empty. A top-level `mcp_tools` list is refused first,
then any untyped `mcp_*` key in `general_settings` or `litellm_settings`, then any `guardrails`,
`policies` or `policy_attachments` (`Error::McpGuardrails`, because the host cannot enforce them), then
missing master-key auth. Each server is then validated (`ConfigFault`) and connected; a server that cannot
be initialized stops the boot, where the Python proxy would start and report it unhealthy.

When serving, tools are published as `<alias or server name, spaces to underscores>-<tool>`, aggregated
over every configured server (there is no per-key scoping: the master key sees everything), with
`_meta["litellm.ai/server_outcomes"]` on `tools/list`. Errors use `{"detail": "<message>"}` and JSON-RPC
code -32003 for forbidden, -32602 for request faults. Caller headers are never copied upstream.

All of this is a pin of the current preview, not the Python contract: where the two differ the
docstring says so, so a change in either direction is a deliberate decision. The binary is only installed
for the CI management group; in the mcp group every test here skips through `gateway_binary()`.
"""

from __future__ import annotations

import base64
import uuid
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.mcp import INITIALIZE, McpPeer, mcp_peer, tool_calls
from pydantic import JsonValue

from tests.integration.configuration.dec1_gwnative_support import (
    GATEWAY_KEY,
    master_key_config,
    native_gateway,
    refused_boot,
)
from tests.integration.mcp.dec1_mcp_support import (
    ACCEPT,
    first_text,
    listed_names,
    rpc,
    rpc_error,
    rpc_result,
)

_FORBIDDEN: Final = "User not allowed to access this MCP server or operation."
_DEAD_URL: Final = "http://127.0.0.1:9/mcp"


def _server(url: str = _DEAD_URL, **fields: JsonValue) -> dict[str, JsonValue]:
    return {"url": url, "transport": "http", **fields}


def _config(servers: Mapping[str, Mapping[str, JsonValue]], **sections: JsonValue) -> dict[str, JsonValue]:
    return master_key_config([], mcp_servers={name: dict(server) for name, server in servers.items()}, **sections)


def _refusal_output(tmp_path: Path, config: dict[str, JsonValue]) -> str:
    refusal: Final = refused_boot(config, tmp_path)
    assert refusal.returncode != 0, refusal.output
    return refusal.output


@contextmanager
def _hosting(
    tmp_path: Path, servers: Mapping[str, Mapping[str, JsonValue]], **sections: JsonValue
) -> Iterator[httpx.Client]:
    with native_gateway(_config(servers, **sections), tmp_path) as gateway, gateway.client() as client:
        yield client


@contextmanager
def _peers(count: int) -> Iterator[tuple[McpPeer, ...]]:
    with ExitStack() as stack:
        yield tuple(stack.enter_context(mcp_peer()) for _ in range(count))


def _call(client: httpx.Client, path: str, name: str, arguments: dict[str, object], **headers: str) -> httpx.Response:
    return rpc(client, path, "tools/call", {"name": name, "arguments": arguments}, key=GATEWAY_KEY, bearer=True, headers=headers)


def _list(client: httpx.Client, path: str = "/mcp", **headers: str) -> httpx.Response:
    return rpc(client, path, "tools/list", key=GATEWAY_KEY, bearer=True, headers=headers)


# Boot refusals ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "section",
    [
        {"policies": {"dec1-policy": {"guardrails": {"add": ["dec1-guard"]}}}},
        {"policy_attachments": [{"policy": "dec1-policy", "scope": "*"}]},
    ],
    ids=["policies", "policy_attachments"],
)
def test_policies_or_policy_attachments_next_to_mcp_servers_refuse_to_boot(
    tmp_path: Path, section: dict[str, JsonValue]
) -> None:
    """One MCP server plus a non-empty `policies` map or `policy_attachments` list and no `guardrails`. The
    check in `build_mcp` covers all three, so the binary must exit non-zero with `McpGuardrails` in its
    output. `test_dec1_gwnative_boot_and_health.py` pins only the `guardrails` arm. UNVERIFIED: the exact
    YAML shapes `litellm_config` accepts for `policies` and `policy_attachments`; if they fail to parse the
    output names a parse error instead."""
    output: Final = _refusal_output(tmp_path, _config({"dec1": _server()}, **section))
    assert "McpGuardrails" in output, output


@pytest.mark.parametrize("mode", ["pre_mcp_call", "during_mcp_call", "post_call"])
def test_any_guardrail_mode_refuses_the_mcp_host_even_when_not_default_on(tmp_path: Path, mode: str) -> None:
    """A single guardrail with `mode` set to an MCP mode or an LLM-only mode, and `default_on` false. The
    host refuses on the presence of any guardrail, not on whether it would apply to MCP, so each must exit
    non-zero with `McpGuardrails`. The Python proxy instead runs `pre_mcp_call` guardrails on tool calls."""
    guardrail: Final = {
        "guardrail_name": "dec1-mode",
        "litellm_params": {"guardrail": "custom_code", "mode": mode, "default_on": False, "custom_code": "x"},
    }
    output: Final = _refusal_output(tmp_path, _config({"dec1": _server()}, guardrails=[guardrail]))
    assert "McpGuardrails" in output, output


def test_mcp_tools_is_refused_before_guardrails_are_considered(tmp_path: Path) -> None:
    """A config with `mcp_tools`, an MCP server and a guardrail. `build_mcp` checks `mcp_tools` first, so
    the output must name `McpSetting("mcp_tools")` and not `McpGuardrails`. Pinned so the reported reason
    stays stable when several apply."""
    guardrail: Final = {"guardrail_name": "dec1-g", "litellm_params": {"guardrail": "presidio", "mode": "pre_call"}}
    config: Final = {**_config({"dec1": _server()}, guardrails=[guardrail]), "mcp_tools": [{"name": "dec1_tool"}]}
    output: Final = _refusal_output(tmp_path, config)
    assert 'McpSetting("mcp_tools")' in output and "McpGuardrails" not in output, output


def test_an_untyped_mcp_setting_is_refused_before_guardrails(tmp_path: Path) -> None:
    """`litellm_settings.mcp_dec1_flag: true` plus a guardrail and one server. The `mcp_*` scan runs before
    the guardrail check, so the output names `McpSetting("mcp_dec1_flag")`."""
    guardrail: Final = {"guardrail_name": "dec1-g", "litellm_params": {"guardrail": "presidio", "mode": "pre_call"}}
    config: Final = _config({"dec1": _server()}, guardrails=[guardrail], litellm_settings={"mcp_dec1_flag": True})
    output: Final = _refusal_output(tmp_path, config)
    assert 'McpSetting("mcp_dec1_flag")' in output, output


def test_typed_mcp_allowed_hosts_and_origins_are_not_refused_as_unknown_mcp_settings(tmp_path: Path) -> None:
    """`general_settings.mcp_allowed_hosts` and `mcp_allowed_origins` are typed settings
    (`config/src/settings.rs`), so they never trip the `mcp_*` scan: with a live peer the gateway boots and
    `/health/readiness` is served."""
    with mcp_peer() as peer, native_gateway(
        _config(
            {"dec1": _server(peer.url)},
            general_settings={"mcp_allowed_hosts": ["127.0.0.1"], "mcp_allowed_origins": []},
        ),
        tmp_path,
    ) as gateway:
        assert gateway.get("/health/readiness").status_code == 200


def test_an_empty_mcp_servers_map_with_guardrails_boots_without_an_mcp_route(tmp_path: Path) -> None:
    """`mcp_servers: {}` and a guardrail. MCP stays off (`build_mcp` returns `None` before the guardrail
    check), so the binary boots and `/mcp-rest/tools/list` with the master key is a plain 404."""
    guardrail: Final = {"guardrail_name": "dec1-g", "litellm_params": {"guardrail": "presidio", "mode": "pre_call"}}
    with native_gateway(master_key_config([], mcp_servers={}, guardrails=[guardrail]), tmp_path) as gateway:
        assert gateway.get("/mcp-rest/tools/list", key=GATEWAY_KEY).status_code == 404


@pytest.mark.parametrize(
    ("server", "fault"),
    [
        ({"url": _DEAD_URL, "transport": "sse"}, "SseTransport"),
        ({"url": _DEAD_URL, "mcp_info": {"mcp_server_cost_info": {"default_cost_per_query": 0.1}}}, "UnsupportedMcpInfo"),
        ({"url": _DEAD_URL, "mcp_info": {"protocol_version": "2025-06-18"}}, "UnsupportedMcpInfo"),
        ({"url": _DEAD_URL, "auth_type": "oauth2", "authentication_token": "x"}, "UnsupportedAuth"),
        ({"url": _DEAD_URL, "command": "python3"}, "HttpShape"),
        ({"transport": "http"}, "HttpShape"),
        ({"transport": "stdio", "command": "python3", "url": _DEAD_URL}, "StdioShape"),
        ({"transport": "stdio", "command": "python3", "static_headers": {"x-a": "b"}}, "StdioCredentials"),
        ({"url": _DEAD_URL, "authentication_token": "orphan"}, "TokenWithoutAuth"),
        ({"url": _DEAD_URL, "auth_type": "bearer_token"}, "AuthWithoutToken"),
        ({"url": _DEAD_URL, "auth_type": "api_key", "authentication_token": "   "}, "EmptyCredential"),
        ({"url": _DEAD_URL, "static_headers": {"content-type": "text/plain"}}, "FramingHeader"),
        ({"url": _DEAD_URL, "static_headers": {"mcp-session-id": "fixed"}}, "FramingHeader"),
        ({"url": "http://user:pw@127.0.0.1:9/mcp"}, "UrlShape"),
        ({"url": "ftp://127.0.0.1/mcp"}, "UrlShape"),
        ({"url": _DEAD_URL, "timeout": 0}, "ZeroTimeout"),
        ({"url": _DEAD_URL, "max_concurrent_requests": 0}, "ConcurrencyOutOfRange"),
        ({"url": _DEAD_URL, "alias": "  "}, "BlankIdentifier"),
    ],
)
def test_a_server_the_rust_host_cannot_honour_refuses_to_boot_with_its_config_fault(
    tmp_path: Path, server: dict[str, JsonValue], fault: str
) -> None:
    """One server whose shape `configured.rs::validate` (or `headers`/`connect_service`) rejects. The boot
    must stop with a non-zero exit and the `ConfigFault` variant name in the output, before any upstream is
    contacted. Several of these are valid on the Python proxy (SSE upstreams, OAuth2, cost info, protocol
    pinning), so each is a feature gap the migration must close or document. UNVERIFIED: the output is
    assumed to carry the `Debug` of `ConnectError::Configuration`, which includes the fault variant name
    (`configuration_faults_reach_the_startup_diagnostic` in `error.rs`)."""
    output: Final = _refusal_output(tmp_path, _config({"dec1": server}))
    assert fault in output, output


@pytest.mark.parametrize("setting", ["allow_all_keys", "disallowed_tools", "allowed_params", "mcp_access_groups"])
def test_python_only_server_settings_are_refused_as_unsupported(tmp_path: Path, setting: str) -> None:
    """A server carrying a Python-proxy setting the Rust `McpServer` struct does not type lands in its
    `unsupported` map, and `validate` refuses it with `UnsupportedSetting("<name>")`. Notably
    `allow_all_keys`, which every Python config-declared server needs for master-key access, blocks the Rust
    boot, so the same config file cannot serve both."""
    value: Final[JsonValue] = {
        "allow_all_keys": True,
        "disallowed_tools": ["fail"],
        "allowed_params": {"add": ["a"]},
        "mcp_access_groups": ["dec1"],
    }[setting]
    output: Final = _refusal_output(tmp_path, _config({"dec1": _server(**{setting: value})}))
    assert "UnsupportedSetting" in output and setting in output, output


def test_two_servers_publishing_the_same_prefix_refuse_to_boot(tmp_path: Path) -> None:
    """Server `dec1a` has `alias: shared` and server `dec1b` has `alias: shared`. `Registry::new` refuses
    duplicate prefixes, so the output must contain `DuplicateIdentifier`. UNVERIFIED: both peers are live so
    the check runs after connection; with dead URLs the boot would fail on `Initialize` first."""
    with mcp_peer() as first, mcp_peer() as second:
        config: Final = _config(
            {"dec1a": _server(first.url, alias="shared"), "dec1b": _server(second.url, alias="shared")}
        )
        output: Final = _refusal_output(tmp_path, config)
    assert "DuplicateIdentifier" in output, output


def test_an_alias_that_equals_another_servers_name_refuses_to_boot(tmp_path: Path) -> None:
    """Server `dec1x` and server `dec1y` with `alias: dec1x`. `Registry::new` also refuses an alias matching
    a previous server's name, so the output contains `DuplicateIdentifier`."""
    with mcp_peer() as first, mcp_peer() as second:
        output: Final = _refusal_output(
            tmp_path, _config({"dec1x": _server(first.url), "dec1y": _server(second.url, alias="dec1x")})
        )
    assert "DuplicateIdentifier" in output, output


def test_an_unreachable_upstream_stops_the_boot(tmp_path: Path) -> None:
    """One well-formed server whose URL refuses connections. The Rust host connects every server at startup
    and fails the boot (`ConnectError::Initialize`, or `Timeout`), where the Python proxy boots and reports
    the server unhealthy. The output must name the server and must not echo the URL's query secret
    (`ConnectError`'s `Debug` redacts the SDK cause)."""
    output: Final = _refusal_output(
        tmp_path, _config({"dec1dead": _server("http://127.0.0.1:9/mcp?token=dec1-private-secret", timeout=3)})
    )
    assert "dec1dead" in output, output
    assert "Initialize" in output or "Timeout" in output, output
    assert "dec1-private-secret" not in output, output


# Serving ------------------------------------------------------------------------------------------------


def test_mcp_routes_require_the_master_key(tmp_path: Path) -> None:
    """With one live server, `tools/list` on `/mcp` without credentials and `GET /mcp-rest/tools/list` with a
    wrong bearer must both be HTTP 401, and the peer must see no `tools/list` from either."""
    with mcp_peer() as peer, _hosting(tmp_path, {"dec1": _server(peer.url)}) as client:
        peer.drain()
        anonymous: Final = rpc(client, "/mcp", "tools/list", key=None)
        wrong: Final = client.get("/mcp-rest/tools/list", headers={"Authorization": "Bearer sk-dec1-wrong"})
        assert anonymous.status_code == 401, anonymous.text
        assert wrong.status_code == 401, wrong.text
        assert not any(item["body"].get("method") == "tools/list" for item in peer.drain() if isinstance(item.get("body"), dict))


def test_mcp_enabled_reports_true(tmp_path: Path) -> None:
    """`GET /mcp/enabled` with the master key must answer HTTP 200 `{"enabled": true}`, the same body the
    Python `get_mcp_server_enabled` route gives when the MCP SDK is installed."""
    with mcp_peer() as peer, _hosting(tmp_path, {"dec1": _server(peer.url)}) as client:
        response: Final = client.get("/mcp/enabled", headers={"Authorization": f"Bearer {GATEWAY_KEY}"})
    assert response.status_code == 200, response.text
    assert response.json() == {"enabled": True}


def test_tools_are_published_under_the_server_name_when_there_is_no_alias(tmp_path: Path) -> None:
    """Server `dec1plain` has no alias. `tools/list` on `/mcp` must name its tools `dec1plain-add`,
    `dec1plain-multiply` and `dec1plain-fail`, matching the Python prefix rule."""
    with mcp_peer() as peer, _hosting(tmp_path, {"dec1plain": _server(peer.url)}) as client:
        names: Final = listed_names(_list(client))
    assert sorted(names) == ["dec1plain-add", "dec1plain-fail", "dec1plain-multiply"], names


def test_an_alias_with_spaces_is_published_with_underscores(tmp_path: Path) -> None:
    """Server `dec1s` has `alias: "dec1 spaced"`. The prefix replaces spaces with underscores
    (`Server::prefix`), so `dec1_spaced-add` must be listed and a call to it must return `5`."""
    with mcp_peer() as peer, _hosting(tmp_path, {"dec1s": _server(peer.url, alias="dec1 spaced")}) as client:
        assert "dec1_spaced-add" in listed_names(_list(client))
        assert first_text(rpc_result(_call(client, "/mcp", "dec1_spaced-add", {"a": 2, "b": 3}))) == "5"


def test_tools_list_carries_server_outcomes_keyed_by_prefix(tmp_path: Path) -> None:
    """Two live servers `dec1a` and `dec1b`. The `tools/list` result must carry
    `_meta["litellm.ai/server_outcomes"]` equal to `{"dec1a": {"status": "ok", "tool_count": 3},
    "dec1b": {"status": "ok", "tool_count": 3}}`, the same wire shape the Python gateway uses."""
    with _peers(2) as (first, second), _hosting(tmp_path, {"dec1a": _server(first.url), "dec1b": _server(second.url)}) as client:
        result: Final = rpc_result(_list(client))
    assert result["_meta"]["litellm.ai/server_outcomes"] == {
        "dec1a": {"status": "ok", "tool_count": 3},
        "dec1b": {"status": "ok", "tool_count": 3},
    }, result


def test_a_prefixed_call_reaches_only_its_server_with_the_bare_name(tmp_path: Path) -> None:
    """Two servers with the same tools. A call to `dec1b-multiply` with `{a: 6, b: 7}` must return `42`;
    only the second peer may receive a `tools/call`, and its params name must be the bare `multiply`."""
    with _peers(2) as (first, second), _hosting(tmp_path, {"dec1a": _server(first.url), "dec1b": _server(second.url)}) as client:
        first.drain()
        second.drain()
        assert first_text(rpc_result(_call(client, "/mcp", "dec1b-multiply", {"a": 6, "b": 7}))) == "42"
        assert tool_calls(first.drain()) == ()
        calls: Final = tool_calls(second.drain())
        assert [call["body"]["params"]["name"] for call in calls] == ["multiply"], calls


def test_a_bare_tool_name_unique_across_servers_is_routed(tmp_path: Path) -> None:
    """With one server, `tools/call` for the bare name `add` matches exactly one catalog entry
    (`request.name == entry.tool.name`), so it must return `5`. Pinned because the Python gateway only keeps
    bare routes under specific discovery conditions (`test_pagination.py`)."""
    with mcp_peer() as peer, _hosting(tmp_path, {"dec1one": _server(peer.url)}) as client:
        assert first_text(rpc_result(_call(client, "/mcp", "add", {"a": 2, "b": 3}))) == "5"


def test_a_bare_tool_name_shared_by_two_servers_is_forbidden_not_guessed(tmp_path: Path) -> None:
    """Two servers both expose `add`. A bare `add` call matches two entries, so the host must answer a
    JSON-RPC error with code -32003 and message `User not allowed to access this MCP server or operation.`
    and neither peer may receive a `tools/call`."""
    with _peers(2) as (first, second), _hosting(tmp_path, {"dec1a": _server(first.url), "dec1b": _server(second.url)}) as client:
        first.drain()
        second.drain()
        error: Final = rpc_error(_call(client, "/mcp", "add", {"a": 1, "b": 1}))
        assert error["code"] == -32003 and error["message"] == _FORBIDDEN, error
        assert tool_calls(first.drain()) == () and tool_calls(second.drain()) == ()


def test_an_unknown_tool_name_is_forbidden(tmp_path: Path) -> None:
    """`tools/call` for `dec1a-nosuch` matches no catalog entry and must answer JSON-RPC -32003 with the
    forbidden message, never forwarding the call. The Python gateway reports an unknown tool differently
    (see `test_dec1_mcp_naming.py`)."""
    with mcp_peer() as peer, _hosting(tmp_path, {"dec1a": _server(peer.url)}) as client:
        peer.drain()
        error: Final = rpc_error(_call(client, "/mcp", "dec1a-nosuch", {}))
        assert error["code"] == -32003, error
        assert tool_calls(peer.drain()) == ()


def test_a_tool_that_raises_upstream_comes_back_as_an_is_error_result(tmp_path: Path) -> None:
    """The peer's `fail` tool raises `synthetic tool failure`. The SDK peer returns an `isError` result, which
    the host relays as a JSON-RPC success whose result has `isError: true` and text containing
    `synthetic tool failure`; it is never turned into a JSON-RPC error."""
    with mcp_peer() as peer, _hosting(tmp_path, {"dec1a": _server(peer.url)}) as client:
        result: Final = rpc_result(_call(client, "/mcp", "dec1a-fail", {}))
    assert result["isError"] is True, result
    assert "synthetic tool failure" in str(first_text(result)), result


def test_a_tools_list_with_a_cursor_is_refused(tmp_path: Path) -> None:
    """The host drains upstream pages itself and publishes one page, so `tools/list` with
    `params.cursor = "x"` must answer JSON-RPC -32602 with message `Gateway listings do not accept an upstream
    cursor`. The Python gateway issues its own portable cursors (`test_pagination.py`)."""
    with mcp_peer() as peer, _hosting(tmp_path, {"dec1a": _server(peer.url)}) as client:
        error: Final = rpc_error(rpc(client, "/mcp", "tools/list", {"cursor": "x"}, key=GATEWAY_KEY, bearer=True))
    assert error["code"] == -32602, error
    assert error["message"] == "Gateway listings do not accept an upstream cursor", error


def test_config_allowed_tools_hides_and_refuses_other_tools(tmp_path: Path) -> None:
    """Server `dec1a` has `allowed_tools: [add]`. `tools/list` must list only `dec1a-add`, and a call to
    `dec1a-multiply` must be JSON-RPC -32003 without reaching the peer."""
    with mcp_peer() as peer, _hosting(tmp_path, {"dec1a": _server(peer.url, allowed_tools=["add"])}) as client:
        assert listed_names(_list(client)) == ["dec1a-add"]
        peer.drain()
        assert rpc_error(_call(client, "/mcp", "dec1a-multiply", {"a": 1, "b": 1}))["code"] == -32003
        assert tool_calls(peer.drain()) == ()


def test_the_server_scoped_path_lists_only_that_server(tmp_path: Path) -> None:
    """Two servers. `tools/list` on `/dec1b/mcp` must list only `dec1b-` tools, and the same on
    `/mcp/dec1b` (the alternate scoped route)."""
    with _peers(2) as (first, second), _hosting(tmp_path, {"dec1a": _server(first.url), "dec1b": _server(second.url)}) as client:
        for path in ("/dec1b/mcp", "/mcp/dec1b"):
            names = listed_names(_list(client, path))
            assert names and all(name.startswith("dec1b-") for name in names), (path, names)


def test_a_scoped_path_naming_an_unknown_server_is_forbidden(tmp_path: Path) -> None:
    """`tools/list` on `/dec1nosuch/mcp` names no configured server, so `servers()` refuses with
    `Error::Forbidden`: HTTP 403 with `{"detail": "User not allowed to access this MCP server or
    operation."}`, or a JSON-RPC -32003 if the SDK transport wraps it. UNVERIFIED which of the two the
    streamable HTTP service surfaces; the test accepts either."""
    with mcp_peer() as peer, _hosting(tmp_path, {"dec1a": _server(peer.url)}) as client:
        response: Final = _list(client, "/dec1nosuch/mcp")
    if response.status_code == 403:
        assert response.json() == {"detail": _FORBIDDEN}
    else:
        assert rpc_error(response)["code"] == -32003, response.text


def test_x_mcp_servers_narrows_the_aggregate_listing(tmp_path: Path) -> None:
    """Two servers. `tools/list` on `/mcp` with `x-mcp-servers: dec1a` must list only `dec1a-` tools; with
    `x-mcp-servers: dec1a,dec1nosuch` the request is forbidden because every named server must exist."""
    with _peers(2) as (first, second), _hosting(tmp_path, {"dec1a": _server(first.url), "dec1b": _server(second.url)}) as client:
        names: Final = listed_names(_list(client, **{"x-mcp-servers": "dec1a"}))
        assert names and all(name.startswith("dec1a-") for name in names), names
        mixed: Final = _list(client, **{"x-mcp-servers": "dec1a,dec1nosuch"})
        assert mixed.status_code == 403 or rpc_error(mixed)["code"] == -32003, mixed.text


def test_caller_headers_and_the_master_key_never_reach_the_upstream(tmp_path: Path) -> None:
    """The caller sends `x-dec1-probe: leak` and its master-key bearer while calling `dec1a-add`. Outbound
    headers are not copied (`gateway-mcp/AGENTS.md`), so no request the peer saw may carry `x-dec1-probe`
    or an `authorization` header containing the master key."""
    with mcp_peer() as peer, _hosting(tmp_path, {"dec1a": _server(peer.url)}) as client:
        peer.drain()
        assert first_text(rpc_result(_call(client, "/mcp", "dec1a-add", {"a": 2, "b": 3}, **{"x-dec1-probe": "leak"}))) == "5"
        seen: Final = peer.drain()
    assert seen, "the peer saw no request"
    for item in seen:
        headers = {str(name).lower(): str(value) for name, value in dict(item.get("headers", {})).items()}
        assert "x-dec1-probe" not in headers, headers
        assert GATEWAY_KEY not in headers.get("authorization", ""), headers


@pytest.mark.parametrize(
    ("auth_type", "token", "slot", "expected"),
    [
        ("api_key", "dec1-token", "x-api-key", "dec1-token"),
        ("bearer_token", "dec1-token", "authorization", "Bearer dec1-token"),
        ("bearer_token", "bEaReR dec1-token", "authorization", "Bearer dec1-token"),
        ("token", "dec1-token", "authorization", "token dec1-token"),
        ("basic", "user:pw", "authorization", "Basic " + base64.b64encode(b"user:pw").decode()),
        ("authorization", "Custom dec1-token", "authorization", "Custom dec1-token"),
    ],
)
def test_configured_upstream_credentials_reach_the_peer_in_their_mode_shape(
    tmp_path: Path, auth_type: str, token: str, slot: str, expected: str
) -> None:
    """Server `dec1a` is configured with `auth_type` and `authentication_token`. Every request the peer sees
    (initialize at boot and the later tool call) must carry header `<slot>` equal to `<expected>`
    (`configured.rs::headers`: bearer normalizes a case-insensitive `Bearer ` prefix, basic base64-encodes a
    raw `user:pw`). The Python proxy uses the same shapes (`test_mcp_credentials.py`)."""
    with mcp_peer() as peer, _hosting(
        tmp_path, {"dec1a": _server(peer.url, auth_type=auth_type, authentication_token=token)}
    ) as client:
        assert first_text(rpc_result(_call(client, "/mcp", "dec1a-add", {"a": 2, "b": 3}))) == "5"
        calls: Final = tool_calls(peer.drain())
    assert calls, "no tools/call reached the peer"
    headers: Final = {str(name).lower(): str(value) for name, value in dict(calls[-1]["headers"]).items()}
    assert headers.get(slot) == expected, headers


def test_upstream_token_header_moves_the_credential_and_overrides_a_static_header(tmp_path: Path) -> None:
    """`auth_type: api_key`, `authentication_token: dec1-token`, `upstream_token_header: x-dec1-auth` and
    `static_headers: {X-Dec1-Auth: old, x-dec1-extra: extra}`. The peer must see `x-dec1-auth: dec1-token`
    (the credential wins over the static header of the same name), `x-dec1-extra: extra`, and no
    `x-api-key`."""
    server: Final = _server(
        auth_type="api_key",
        authentication_token="dec1-token",
        upstream_token_header="x-dec1-auth",
        static_headers={"X-Dec1-Auth": "old", "x-dec1-extra": "extra"},
    )
    with mcp_peer() as peer:
        with _hosting(tmp_path, {"dec1a": {**server, "url": peer.url}}) as client:
            assert first_text(rpc_result(_call(client, "/mcp", "dec1a-add", {"a": 2, "b": 3}))) == "5"
        calls: Final = tool_calls(peer.drain())
    headers: Final = {str(name).lower(): str(value) for name, value in dict(calls[-1]["headers"]).items()}
    assert headers.get("x-dec1-auth") == "dec1-token", headers
    assert headers.get("x-dec1-extra") == "extra", headers
    assert "x-api-key" not in headers, headers


def test_rest_listing_returns_bare_names_with_mcp_info_and_a_success_message(tmp_path: Path) -> None:
    """`GET /mcp-rest/tools/list` with the master key and one server `dec1a` (alias `dec1alias`,
    `server_id: dec1-fixed-id`). Each entry must have the bare tool name (`add`, not `dec1alias-add`) and
    `mcp_info == {"server_id": "dec1-fixed-id", "server_name": "dec1a", "alias": "dec1alias"}`; the body has
    `error: null` and `message: "Successfully retrieved tools"`."""
    with mcp_peer() as peer, _hosting(
        tmp_path, {"dec1a": _server(peer.url, alias="dec1alias", server_id="dec1-fixed-id")}
    ) as client:
        response: Final = client.get("/mcp-rest/tools/list", headers={"Authorization": f"Bearer {GATEWAY_KEY}"})
    assert response.status_code == 200, response.text
    body: Final = response.json()
    assert body["error"] is None and body["message"] == "Successfully retrieved tools", body
    assert sorted(tool["name"] for tool in body["tools"]) == ["add", "fail", "multiply"], body
    assert {tuple(sorted(tool["mcp_info"].items())) for tool in body["tools"]} == {
        (("alias", "dec1alias"), ("server_id", "dec1-fixed-id"), ("server_name", "dec1a"))
    }, body


def test_a_server_without_a_configured_id_gets_a_stable_32_hex_id(tmp_path: Path) -> None:
    """With no `server_id`, `info()` derives one from a SHA-256 of name, url, transport, auth type and alias
    and keeps the first 32 hex characters. Two boots of the same config must report the same 32 character
    lowercase hex `mcp_info.server_id`, so saved REST `server_id`s survive restarts."""
    seen: list[str] = []
    with mcp_peer() as peer:
        for attempt in range(2):
            directory = tmp_path / str(attempt)
            directory.mkdir()
            with _hosting(directory, {"dec1a": _server(peer.url)}) as client:
                body = client.get("/mcp-rest/tools/list", headers={"Authorization": f"Bearer {GATEWAY_KEY}"}).json()
                seen.append(str(body["tools"][0]["mcp_info"]["server_id"]))
    assert seen[0] == seen[1], seen
    assert len(seen[0]) == 32 and all(char in "0123456789abcdef" for char in seen[0]), seen


def test_rest_call_without_server_id_is_400_with_a_flat_detail(tmp_path: Path) -> None:
    """`POST /mcp-rest/tools/call` with `{"name": "add"}` must be HTTP 400 with exactly
    `{"detail": "server_id is required in request body"}`. The Python gateway nests it as
    `detail.error/detail.message` (`test_dec1_mcp_allowlists.py`)."""
    with mcp_peer() as peer, _hosting(tmp_path, {"dec1a": _server(peer.url)}) as client:
        response: Final = client.post(
            "/mcp-rest/tools/call", json={"name": "add"}, headers={"Authorization": f"Bearer {GATEWAY_KEY}"}
        )
    assert response.status_code == 400, response.text
    assert response.json() == {"detail": "server_id is required in request body"}


def test_rest_call_by_server_name_returns_the_tool_result(tmp_path: Path) -> None:
    """`POST /mcp-rest/tools/call` with `{"server_id": "dec1a", "name": "add", "arguments": {a: 2, b: 3}}`
    (the server name stands in for the id, `Server::matches`) must return HTTP 200 with
    `content[0].text == "5"` and `isError: false`."""
    with mcp_peer() as peer, _hosting(tmp_path, {"dec1a": _server(peer.url)}) as client:
        response: Final = client.post(
            "/mcp-rest/tools/call",
            json={"server_id": "dec1a", "name": "add", "arguments": {"a": 2, "b": 3}},
            headers={"Authorization": f"Bearer {GATEWAY_KEY}"},
        )
    assert response.status_code == 200, response.text
    assert response.json()["content"][0]["text"] == "5" and response.json()["isError"] is False, response.text


def test_rest_listing_rejects_an_unknown_query_parameter(tmp_path: Path) -> None:
    """`ListQuery` denies unknown fields, so `GET /mcp-rest/tools/list?dec1=1` must be HTTP 400 with
    `detail` starting `invalid query string`. The Python route ignores unknown parameters."""
    with mcp_peer() as peer, _hosting(tmp_path, {"dec1a": _server(peer.url)}) as client:
        response: Final = client.get(
            "/mcp-rest/tools/list", params={"dec1": "1"}, headers={"Authorization": f"Bearer {GATEWAY_KEY}"}
        )
    assert response.status_code == 400, response.text
    assert str(response.json()["detail"]).startswith("invalid query string"), response.text


def test_an_initialize_returns_a_session_id_and_an_unknown_session_is_404(tmp_path: Path) -> None:
    """`initialize` on `/mcp` without a session id goes to the stateful service and must return an
    `mcp-session-id`. A `tools/list` with a made-up session id must be HTTP 404
    `{"detail": "MCP session not found"}`; the Python gateway strips the stale id and serves the request."""
    with mcp_peer() as peer, _hosting(tmp_path, {"dec1a": _server(peer.url)}) as client:
        initialized: Final = rpc(client, "/mcp", "initialize", INITIALIZE, key=GATEWAY_KEY, bearer=True)
        assert initialized.status_code == 200 and initialized.headers.get("mcp-session-id"), initialized.text
        stale: Final = _list(client, **{"mcp-session-id": uuid.uuid4().hex})
    assert stale.status_code == 404, stale.text
    assert stale.json() == {"detail": "MCP session not found"}


def test_a_deleted_session_cannot_be_reused(tmp_path: Path) -> None:
    """The owner initializes, then `DELETE /mcp` with its session id succeeds and the owner map forgets it,
    so a following `tools/list` with that id is HTTP 404 `MCP session not found`, and a second DELETE is
    also 404 (the Python gateway answers 200 `Session terminated successfully`)."""
    with mcp_peer() as peer, _hosting(tmp_path, {"dec1a": _server(peer.url)}) as client:
        initialized: Final = rpc(client, "/mcp", "initialize", INITIALIZE, key=GATEWAY_KEY, bearer=True)
        session: Final = initialized.headers["mcp-session-id"]
        headers: Final = {"Authorization": f"Bearer {GATEWAY_KEY}", "mcp-session-id": session, "Accept": ACCEPT}
        assert client.delete("/mcp", headers=headers).status_code in (200, 202, 204)
        assert _list(client, **{"mcp-session-id": session}).status_code == 404
        again: Final = client.delete("/mcp", headers=headers)
    assert again.status_code == 404 and again.json() == {"detail": "MCP session not found"}, again.text


def test_a_session_bound_on_the_aggregate_path_is_foreign_on_a_scoped_path(tmp_path: Path) -> None:
    """A session opened on `/mcp` is bound to a fingerprint of the credential and the requested server
    scope. Presenting it on `/dec1a/mcp` must be refused with HTTP 403 `{"detail": "MCP session belongs to
    another caller"}`. UNVERIFIED: whether the gateway's `bind_session_owner` middleware inserts a
    `SessionOwner` principal that drops the scope from the fingerprint; if it does, the request is served
    and this test must change to expect 200."""
    with mcp_peer() as peer, _hosting(tmp_path, {"dec1a": _server(peer.url)}) as client:
        initialized: Final = rpc(client, "/mcp", "initialize", INITIALIZE, key=GATEWAY_KEY, bearer=True)
        session: Final = initialized.headers["mcp-session-id"]
        response: Final = _list(client, "/dec1a/mcp", **{"mcp-session-id": session})
    assert response.status_code == 403, response.text
    assert response.json() == {"detail": "MCP session belongs to another caller"}


def test_an_aggregate_resource_read_with_two_servers_is_refused_as_ambiguous(tmp_path: Path) -> None:
    """Two rich servers. `resources/read` for `status://ready` on `/mcp` must answer JSON-RPC -32602 with
    message `read_resource requires exactly one allowed MCP server` (or HTTP 400 with that detail), and on
    `/dec1a/mcp` it must return contents with text `ready`. The Python gateway resolves the owner instead
    (#51462)."""
    with mcp_peer(rich=True) as first, mcp_peer(rich=True) as second, _hosting(
        tmp_path, {"dec1a": _server(first.url), "dec1b": _server(second.url)}
    ) as client:
        aggregate: Final = rpc(client, "/mcp", "resources/read", {"uri": "status://ready"}, key=GATEWAY_KEY, bearer=True)
        scoped: Final = rpc(client, "/dec1a/mcp", "resources/read", {"uri": "status://ready"}, key=GATEWAY_KEY, bearer=True)
    if aggregate.status_code == 400:
        assert aggregate.json() == {"detail": "read_resource requires exactly one allowed MCP server"}
    else:
        error = rpc_error(aggregate)
        assert error["code"] == -32602 and error["message"] == "read_resource requires exactly one allowed MCP server", error
    assert rpc_result(scoped)["contents"][0]["text"] == "ready", scoped.text


def test_prompts_and_resources_are_prefixed_but_resource_uris_are_not(tmp_path: Path) -> None:
    """One rich server `dec1r`. `prompts/list` must contain a prompt named `dec1r-greeting`, and
    `resources/list` a resource named `dec1r-status` whose `uri` is still `status://ready` (names get the
    prefix, URIs do not). UNVERIFIED: the SDK names the resource after the function, `status`."""
    with mcp_peer(rich=True) as peer, _hosting(tmp_path, {"dec1r": _server(peer.url)}) as client:
        prompts: Final = rpc_result(rpc(client, "/mcp", "prompts/list", key=GATEWAY_KEY, bearer=True))
        resources: Final = rpc_result(rpc(client, "/mcp", "resources/list", key=GATEWAY_KEY, bearer=True))
    assert "dec1r-greeting" in [prompt["name"] for prompt in prompts["prompts"]], prompts
    by_uri: Final = {resource["uri"]: resource["name"] for resource in resources["resources"]}
    assert by_uri.get("status://ready") == "dec1r-status", resources


def test_a_request_with_a_foreign_host_header_is_refused(tmp_path: Path) -> None:
    """With the default allowed hosts (`localhost`, `127.0.0.1`, `::1`), a `tools/list` on `/mcp` sent with
    `Host: dec1.example` must be HTTP 403 and never reach the peer (DNS rebinding protection). The Python
    gateway has no host allowlist on `/mcp`. UNVERIFIED: the response body comes from rmcp's streamable HTTP
    service and is not asserted."""
    with mcp_peer() as peer, _hosting(tmp_path, {"dec1a": _server(peer.url)}) as client:
        peer.drain()
        response: Final = _list(client, Host="dec1.example")
        assert response.status_code == 403, response.text
        assert not any(
            isinstance(item.get("body"), dict) and item["body"].get("method") == "tools/list" for item in peer.drain()
        )


def test_the_native_host_serves_mcp_next_to_inference_on_one_port(tmp_path: Path) -> None:
    """The same gateway answers `/health/readiness` with `{"status": "healthy", "db": "Not connected"}` and
    `/mcp-rest/tools/list` with tools, so MCP shares the inference listener and its auth layer rather than a
    second port."""
    with mcp_peer() as peer, native_gateway(_config({"dec1a": _server(peer.url)}), tmp_path) as gateway:
        assert gateway.get("/health/readiness").json() == {"status": "healthy", "db": "Not connected"}
        listed: Final = gateway.get("/mcp-rest/tools/list", key=GATEWAY_KEY)
        assert listed.status_code == 200 and listed.json()["tools"], listed.text
