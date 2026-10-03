import functools
import json
import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Final

import pytest
import yaml
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule, run_state_machine_as_test
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.generation import LIFECYCLE_SETTINGS, bounded_http_requests
from integration._support.mcp import (
    McpCaller,
    Outcome,
    call_tool,
    forget_mcp,
    mcp_peer,
    register_mcp,
    tool_calls,
    tool_names,
)
from integration._support.process import owned_proxy


@pytest.mark.covers("mcp.call_tool.saved_headers.reach_actual_transport")
def test_saved_headers_reach_real_mcp_tool_and_survive_unrelated_edit(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "integration" + uuid.uuid4().hex
        identity: Final = register_mcp(
            scenario, peer, alias, static_headers={"X-Integration-Saved": "synthetic-header-value"}
        )
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        for generation in range(2):
            names: Final = tool_names(gateway, key, identity)
            assert set(names) == {"add", "multiply", "fail"}
            peer.drain()
            response: Final = call_tool(gateway, key, identity, names["add"], {"a": 3, "b": 5})
            assert response.status_code == 200, response.text
            assert response.json()["isError"] is False
            assert len(response.json()["content"]) == 1
            assert response.json()["content"][0]["type"] == "text"
            assert response.json()["content"][0]["text"] == "8"
            calls: Final = tuple(item for item in peer.drain() if item["body"].get("method") == "tools/call")
            assert len(calls) == 1
            assert calls[0]["headers"][b"x-integration-saved"] == b"synthetic-header-value"
            assert calls[0]["body"]["params"]["name"] == "add"
            assert calls[0]["body"]["params"]["arguments"] == {"a": 3, "b": 5}
            if generation == 0:
                updated: Final = gateway.request(
                    "PUT", "/v1/mcp/server", {"server_id": identity, "server_name": alias + "renamed"}
                )
                assert updated.status_code == 202, updated.text
        rows: Final = read_rows('SELECT server_name FROM "LiteLLM_MCPServerTable" WHERE server_id = %s', (identity,))
        assert rows == [{"server_name": alias + "renamed"}]


@pytest.mark.covers("mcp.call_tool.errors.tool_failure_is_not_success")
def test_tool_error_remains_error_and_healthy_sibling_returns_value(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        identity: Final = register_mcp(scenario, peer, "integration" + uuid.uuid4().hex)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        names: Final = tool_names(gateway, key, identity)
        failure: Final = call_tool(gateway, key, identity, names["fail"], {})
        assert failure.status_code == 200, failure.text
        assert failure.json()["isError"] is True
        assert failure.json()["content"][0]["text"] == "Error executing tool fail"
        healthy: Final = call_tool(gateway, key, identity, names["multiply"], {"a": 3, "b": 5})
        assert healthy.status_code == 200, healthy.text
        assert healthy.json()["isError"] is False
        assert healthy.json()["content"][0]["text"] == "15"


@pytest.mark.timeout(180)
@pytest.mark.covers("other.mcp.lifecycle.generated_save_reload_preserves_effective_headers")
def test_generated_mcp_edits_preserve_actual_headers_and_tool_results(gateway: Gateway) -> None:
    with mcp_peer() as peer, bounded_http_requests((gateway,), limit=1500) as budget:

        class Servers(RuleBasedStateMachine):
            def __init__(self) -> None:
                super().__init__()
                self.resources = ExitStack()
                self.marker = "first"
                self.name = "integration" + uuid.uuid4().hex
                try:
                    scenario = self.resources.enter_context(gateway.scenario())
                    self.identity = register_mcp(
                        scenario, peer, self.name, static_headers={"X-Integration-Saved": self.marker}
                    )
                    self.key = scenario.key(object_permission={"mcp_servers": [self.identity]})
                except BaseException:
                    with budget.cleanup():
                        self.resources.close()
                    raise

            @rule(value=st.sampled_from(("first", "second", "third")))
            def header(self, value: str) -> None:
                response: Final = gateway.request(
                    "PUT",
                    "/v1/mcp/server",
                    {"server_id": self.identity, "static_headers": {"X-Integration-Saved": value}},
                )
                assert response.status_code == 202, response.text
                self.marker = value

            @rule(value=st.sampled_from(("original", "renamed")))
            def rename(self, value: str) -> None:
                response: Final = gateway.request(
                    "PUT", "/v1/mcp/server", {"server_id": self.identity, "server_name": self.name + value}
                )
                assert response.status_code == 202, response.text

            @invariant()
            def persisted_configuration_controls_actual_tools(self) -> None:
                names: Final = tool_names(gateway, self.key, self.identity)
                assert set(names) == {"add", "multiply", "fail"}
                peer.drain()
                result: Final = call_tool(gateway, self.key, self.identity, names["add"], {"a": 3, "b": 5})
                assert result.status_code == 200 and result.json()["isError"] is False, result.text
                assert result.json()["content"][0]["text"] == "8"
                calls: Final = tuple(item for item in peer.drain() if item["body"].get("method") == "tools/call")
                assert len(calls) == 1 and calls[0]["headers"][b"x-integration-saved"] == self.marker.encode()
                assert (
                    len(
                        read_rows('SELECT server_id FROM "LiteLLM_MCPServerTable" WHERE server_id=%s', (self.identity,))
                    )
                    == 1
                )

            def teardown(self) -> None:
                with budget.cleanup():
                    self.resources.close()

        run_state_machine_as_test(Servers, settings=LIFECYCLE_SETTINGS)


@pytest.mark.covers("other.mcp.health.restricted_keys_intersect_grants_in_both_modes")
def test_health_intersects_route_restricted_key_grants_in_both_management_modes(
    gateway: Gateway, tmp_path: Path
) -> None:
    for mode in ("restricted", "view_all"):
        config = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["general_settings"]["user_mcp_management_mode"] = mode
        path = tmp_path / f"health-{mode}.yaml"
        path.write_text(yaml.safe_dump(config))
        with (
            owned_proxy(gateway, tmp_path, {}, config=path) as candidate,
            mcp_peer() as peer,
            candidate.scenario() as scenario,
        ):
            first = register_mcp(scenario, peer, "health" + uuid.uuid4().hex)
            second = register_mcp(scenario, peer, "health" + uuid.uuid4().hex)
            control = scenario.key(object_permission={"mcp_servers": [first]})
            names = tool_names(candidate, control, first)
            healthy = call_tool(candidate, control, first, names["add"], {"a": 3, "b": 5})
            assert healthy.status_code == 200 and healthy.json()["content"][0]["text"] == "8", healthy.text
            for grants in ([first], [second], []):
                key = scenario.key(
                    allowed_routes=["/v1/mcp/server", "/v1/mcp/server/health"],
                    object_permission={"mcp_servers": grants},
                )
                listed = candidate.request("GET", "/v1/mcp/server", key=key)
                assert listed.status_code == 200, listed.text
                assert {row["server_id"] for row in listed.json()} == set(grants), listed.text
                for requested in (None, [second], [first, second]):
                    response = candidate.client.get(
                        "/v1/mcp/server/health",
                        headers={"Authorization": f"Bearer {key}"},
                        params=[] if requested is None else [("server_ids", identity) for identity in requested],
                    )
                    assert response.status_code == 200, response.text
                    expected = set(grants) if requested is None else set(grants).intersection(requested)
                    assert {row["server_id"] for row in response.json()} == expected, response.text
                    assert all(row["status"] == "healthy" for row in response.json())


@pytest.mark.covers("other.mcp.credentials.warm_removal_fails_closed_without_upstream_traffic")
def test_warm_credential_removal_rejects_without_upstream_traffic(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        identity = register_mcp(
            scenario,
            peer,
            "credentials" + uuid.uuid4().hex,
            auth_type="bearer_token",
            static_headers={"Authorization": "Bearer synthetic-upstream-credential"},
        )
        key = scenario.key(object_permission={"mcp_servers": [identity]})
        names = tool_names(gateway, key, identity)
        warm = call_tool(gateway, key, identity, names["add"], {"a": 3, "b": 5})
        assert warm.status_code == 200 and warm.json()["content"][0]["text"] == "8", warm.text
        calls = tuple(item for item in peer.drain() if item["body"].get("method") == "tools/call")
        assert len(calls) == 1
        assert calls[0]["headers"][b"authorization"] == b"Bearer synthetic-upstream-credential"
        removed = gateway.request("PUT", "/v1/mcp/server", {"server_id": identity, "static_headers": {}})
        assert removed.status_code == 202, removed.text
        stored = gateway.request("GET", f"/v1/mcp/server/{identity}")
        assert stored.status_code == 200, stored.text
        assert stored.json()["auth_type"] == "bearer_token"
        assert not stored.json().get("static_headers"), stored.text
        peer.drain()
        for operation in ("list", "call"):
            rejected = (
                gateway.client.get(
                    "/mcp-rest/tools/list", params={"server_id": identity}, headers={"x-litellm-api-key": key}
                )
                if operation == "list"
                else call_tool(gateway, key, identity, names["add"], {"a": 3, "b": 5})
            )
            assert rejected.status_code == 500, rejected.text
            if operation == "list":
                assert rejected.json()["detail"]["error"] == "internal", rejected.text
                assert "Failed to list tools from server" in rejected.json()["detail"]["message"], rejected.text
            else:
                assert "requires a usable upstream credential" in rejected.text, rejected.text
            assert peer.drain() == (), "missing static credential escaped to upstream"
        changed = gateway.request(
            "PUT",
            "/v1/mcp/server",
            {
                "server_id": identity,
                "auth_type": "oauth2_token_exchange",
                "token_exchange_endpoint": peer.url + "/token",
                "credentials": {"client_id": "synthetic-client"},
            },
        )
        assert changed.status_code == 202, changed.text
        peer.drain()
        rejected_subject = call_tool(gateway, key, identity, names["add"], {"a": 3, "b": 5})
        assert rejected_subject.status_code == 401, rejected_subject.text
        assert peer.drain() == (), "virtual key cannot supply an OBO subject token"
        control_id = register_mcp(scenario, peer, "control" + uuid.uuid4().hex, auth_type="none")
        control_key = scenario.key(object_permission={"mcp_servers": [control_id]})
        control_names = tool_names(gateway, control_key, control_id)
        control = call_tool(gateway, control_key, control_id, control_names["multiply"], {"a": 3, "b": 5})
        assert control.status_code == 200 and control.json()["content"][0]["text"] == "15", control.text


@pytest.mark.parametrize("authenticated", (False, True), ids=("anonymous", "bearer"))
@pytest.mark.covers("other.mcp.permissions.same_url_servers_enforce_discovery_and_execution")
def test_same_url_server_grants_scope_discovery_and_direct_or_virtual_execution(
    gateway: Gateway, authenticated: bool
) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        aliases: Final = tuple("scope" + uuid.uuid4().hex for _ in range(2))
        servers: Final = tuple(
            register_mcp(
                scenario,
                peer,
                alias,
                auth_type="bearer_token" if authenticated else "none",
                static_headers={
                    "X-Integration-Server": alias,
                    **({"Authorization": f"Bearer synthetic-{alias}"} if authenticated else {}),
                },
            )
            for alias in aliases
        )
        for virtual in (False, True):
            keys = tuple(
                scenario.key(object_permission={"mcp_servers": [server], "mcp_tool_search_enabled": virtual})
                for server in servers
            )
            for server, alias, key in zip(servers, aliases, keys):
                catalog = gateway.request("GET", "/mcp-rest/tools/list", key=key)
                assert catalog.status_code == 200, catalog.text
                if virtual:
                    assert {tool["name"] for tool in catalog.json()["tools"]} == {
                        "mcp_tool_search",
                        "mcp_tool_call",
                        "agent_search",
                        "skill_search",
                    }, catalog.text
                    search = gateway.request(
                        "POST",
                        "/mcp-rest/tools/call",
                        {"name": "mcp_tool_search", "arguments": {"query": "add", "top_k": 10}},
                        key=key,
                    )
                    assert search.status_code == 200 and search.json()["isError"] is False, search.text
                    assert [tool["name"] for tool in json.loads(search.json()["content"][0]["text"])] == [
                        f"{alias}-add"
                    ], search.text
                else:
                    assert {tool["mcp_info"]["server_id"] for tool in catalog.json()["tools"]} == {server}
                    assert {tool["name"] for tool in catalog.json()["tools"]} == {"add", "multiply", "fail"}
            for server_index, caller_index in ((0, 0), (1, 0), (1, 1)):
                peer.drain()
                response = gateway.request(
                    "POST",
                    "/mcp-rest/tools/call",
                    {
                        "name": "mcp_tool_call" if virtual else "add",
                        **({} if virtual else {"server_id": servers[server_index]}),
                        "arguments": (
                            {"tool_name": f"{aliases[server_index]}-add", "arguments": {"a": 3, "b": 5}}
                            if virtual
                            else {"a": 3, "b": 5}
                        ),
                    },
                    key=keys[caller_index],
                )
                observed = peer.drain()
                if server_index != caller_index:
                    assert response.status_code == 403 and "not allowed" in response.text, response.text
                    assert tool_calls(observed) == (), "forbidden server reached the upstream"
                    continue
                assert response.status_code == 200 and response.json()["isError"] is False, response.text
                assert response.json()["content"][0]["text"] == "8", response.text
                calls = tuple(item for item in observed if item["body"].get("method") == "tools/call")
                assert len(calls) == 1
                assert calls[0]["headers"][b"x-integration-server"] == aliases[server_index].encode()
                assert all(
                    item["headers"].get(b"authorization")
                    == (f"Bearer synthetic-{_server_alias(item)}".encode() if authenticated else None)
                    for item in observed
                ), observed


def _matches_grants(expected: set[str], view: Outcome) -> bool:
    return view.error is None and set(view.tools) == expected


def _granted_view(worker: Gateway, key: str) -> Outcome:
    return McpCaller(worker, key, "mcp").list_tools()


def _server_alias(call: dict[str, object]) -> str:
    headers: Final = call["headers"]
    assert isinstance(headers, dict)
    return headers[b"x-integration-server"].decode()


@pytest.mark.timeout(600)
def test_generated_create_edit_grant_revoke_delete_call_keeps_grants_and_tool_lists_consistent(
    gateway: Gateway, peer: Gateway
) -> None:
    with mcp_peer() as upstream, bounded_http_requests((gateway, peer), limit=6000) as budget:

        class Fleet(RuleBasedStateMachine):
            def __init__(self) -> None:
                super().__init__()
                self.resources = ExitStack()
                self.servers: dict[str, str] = {}
                self.grants: dict[str, set[str]] = {}
                self.keys: tuple[str, ...] = ()
                try:
                    self.scenario = self.resources.enter_context(gateway.scenario())
                    self.create()
                    self.keys = tuple(
                        self.scenario.key(object_permission={"mcp_servers": list(self.servers.values())[:count]})
                        for count in (0, 1)
                    )
                    self.grants = {self.keys[0]: set(), self.keys[1]: set(self.servers)}
                except BaseException:
                    with budget.cleanup():
                        self.resources.close()
                    raise

            @rule()
            def create(self) -> None:
                if len(self.servers) >= 3:
                    return
                alias: Final = "fleet" + uuid.uuid4().hex[:8]
                identity: Final = register_mcp(
                    self.scenario,
                    upstream,
                    alias,
                    cleanup=forget_mcp,
                    static_headers={"X-Integration-Server": alias},
                )
                self.servers[alias] = identity

            @rule(index=st.integers(0, 2), suffix=st.sampled_from(("", "renamed")))
            def edit(self, index: int, suffix: str) -> None:
                if not self.servers:
                    return
                alias: Final = sorted(self.servers)[index % len(self.servers)]
                response: Final = gateway.request(
                    "PUT",
                    "/v1/mcp/server",
                    {"server_id": self.servers[alias], "description": alias + suffix, "alias": alias},
                )
                assert response.status_code in (200, 202), response.text

            @rule(key_index=st.integers(0, 1), index=st.integers(0, 2), granted=st.booleans())
            def grant_or_revoke(self, key_index: int, index: int, granted: bool) -> None:
                if not self.servers:
                    return
                previous: Final = self.keys[key_index]
                alias: Final = sorted(self.servers)[index % len(self.servers)]
                wanted: Final = (self.grants[previous] | {alias}) if granted else (self.grants[previous] - {alias})
                key: Final = self.scenario.key(
                    object_permission={"mcp_servers": [self.servers[a] for a in sorted(wanted)]}
                )
                self.keys = tuple(key if i == key_index else k for i, k in enumerate(self.keys))
                del self.grants[previous]
                self.grants[key] = wanted

            @rule(index=st.integers(0, 2))
            def delete(self, index: int) -> None:
                if len(self.servers) <= 1:
                    return
                alias: Final = sorted(self.servers)[index % len(self.servers)]
                response: Final = gateway.request("DELETE", f"/v1/mcp/server/{self.servers[alias]}")
                assert response.status_code in (200, 202), response.text
                del self.servers[alias]
                for key in self.keys:
                    self.grants[key].discard(alias)

            @invariant()
            def tool_lists_and_calls_match_grants_on_both_workers(self) -> None:
                for key in self.keys:
                    expected = {f"{alias}-{tool}" for alias in self.grants[key] for tool in ("add", "multiply", "fail")}
                    for worker in (gateway, peer):
                        listing = eventually(
                            functools.partial(_granted_view, worker, key),
                            functools.partial(_matches_grants, expected),
                            seconds=40,
                            return_last_on_timeout=True,
                        )
                        assert set(listing.tools) == expected, (worker.client.base_url, listing.raw)
                    upstream.drain()
                    caller = McpCaller(gateway, key, "mcp")
                    for alias in self.grants[key]:
                        served = caller.call(f"{alias}-add", {"a": 2, "b": 3})
                        assert served.text == "5", served.raw
                    reached = tool_calls(upstream.drain())
                    assert sorted(_server_alias(call) for call in reached) == sorted(self.grants[key]), reached
                    for alias in set(self.servers) - self.grants[key]:
                        denied = caller.call(f"{alias}-add", {"a": 2, "b": 3})
                        assert denied.error is not None and denied.text != "5", denied.raw
                    assert tool_calls(upstream.drain()) == (), "a revoked or never-granted call reached the peer"

            def teardown(self) -> None:
                with budget.cleanup():
                    self.resources.close()

        run_state_machine_as_test(Fleet, settings=settings(LIFECYCLE_SETTINGS, max_examples=5, stateful_step_count=6))


def test_key_grant_added_by_key_update_is_visible_to_mcp_tool_listing_before_the_cache_ttl(gateway: Gateway) -> None:
    with mcp_peer() as upstream, gateway.scenario() as scenario:
        alias: Final = "late" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, upstream, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": []})
        assert _granted_view(gateway, key).tools == ()
        updated: Final = gateway.request(
            "POST", "/key/update", {"key": key, "object_permission": {"mcp_servers": [identity]}}
        )
        assert updated.status_code == 200, updated.text
        seen: Final = eventually(
            lambda: _granted_view(gateway, key), lambda view: view.tools != (), seconds=15, return_last_on_timeout=True
        )
        assert set(seen.tools) == {f"{alias}-add", f"{alias}-multiply", f"{alias}-fail"}, seen.raw


def _update_tool_permissions(
    gateway: Gateway, key: str, identity: str, permissions: dict[str, list[str]] | None
) -> None:
    updated: Final = gateway.request(
        "POST",
        "/key/update",
        {"key": key, "object_permission": {"mcp_servers": [identity], "mcp_tool_permissions": permissions}},
    )
    assert updated.status_code == 200, updated.text


def _listing_on_both(
    gateway: Gateway, peer: Gateway, key: str, expected: set[str]
) -> None:
    for worker in (gateway, peer):
        listing: Final = eventually(
            functools.partial(_granted_view, worker, key),
            functools.partial(_matches_grants, expected),
            seconds=15,
            return_last_on_timeout=True,
        )
        assert set(listing.tools) == expected, (worker.client.base_url, listing.raw)


def _multiply_outcome_on_both(gateway: Gateway, peer: Gateway, key: str, alias: str) -> tuple[Outcome, Outcome]:
    return (
        McpCaller(gateway, key, "mcp").call(f"{alias}-multiply", {"a": 2, "b": 3}),
        McpCaller(peer, key, "mcp").call(f"{alias}-multiply", {"a": 2, "b": 3}),
    )


def test_key_update_tool_permission_widen_narrow_and_clear_apply_on_both_workers(
    gateway: Gateway, peer: Gateway
) -> None:
    with mcp_peer() as upstream, gateway.scenario() as scenario:
        alias: Final = "perm" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, upstream, alias)
        key: Final = scenario.key(
            object_permission={"mcp_servers": [identity], "mcp_tool_permissions": {identity: ["add"]}}
        )
        add_only: Final = {f"{alias}-add"}
        all_tools: Final = {f"{alias}-add", f"{alias}-multiply", f"{alias}-fail"}
        upstream.drain()

        _listing_on_both(gateway, peer, key, add_only)
        denied: Final = _multiply_outcome_on_both(gateway, peer, key, alias)
        assert all(call.error is not None and call.text != "6" for call in denied), [call.raw for call in denied]
        assert tool_calls(upstream.drain()) == (), "a denied call reached the peer"

        _update_tool_permissions(gateway, key, identity, {identity: ["add", "multiply"]})
        _listing_on_both(gateway, peer, key, {f"{alias}-add", f"{alias}-multiply"})
        widened: Final = _multiply_outcome_on_both(gateway, peer, key, alias)
        assert [call.text for call in widened] == ["6", "6"], [call.raw for call in widened]

        _update_tool_permissions(gateway, key, identity, {identity: ["add"]})
        _listing_on_both(gateway, peer, key, add_only)
        upstream.drain()
        narrowed: Final = _multiply_outcome_on_both(gateway, peer, key, alias)
        assert all(call.error is not None and call.text != "6" for call in narrowed), [call.raw for call in narrowed]
        assert tool_calls(upstream.drain()) == (), "a revoked call reached the peer"

        _update_tool_permissions(gateway, key, identity, {})
        _listing_on_both(gateway, peer, key, all_tools)
        cleared: Final = _multiply_outcome_on_both(gateway, peer, key, alias)
        assert [call.text for call in cleared] == ["6", "6"], [call.raw for call in cleared]

        _update_tool_permissions(gateway, key, identity, {identity: ["add"]})
        _listing_on_both(gateway, peer, key, add_only)

        _update_tool_permissions(gateway, key, identity, None)
        _listing_on_both(gateway, peer, key, all_tools)
        nulled: Final = _multiply_outcome_on_both(gateway, peer, key, alias)
        assert [call.text for call in nulled] == ["6", "6"], [call.raw for call in nulled]
