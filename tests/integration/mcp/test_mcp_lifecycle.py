import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Final

import pytest
import yaml
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule, run_state_machine_as_test

from integration._support.client import Gateway
from integration._support.database import read_rows
from integration._support.generation import LIFECYCLE_SETTINGS, bounded_http_requests
from integration._support.process import owned_proxy
from integration._support.mcp import call_tool, mcp_peer, register_mcp, tool_names


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
        assert "synthetic tool failure" in failure.json()["content"][0]["text"]
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
            owned = {first, second}
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
                assert {row["server_id"] for row in listed.json()}.intersection(owned) == set(grants)
                for requested in (None, [second], [first, second]):
                    response = candidate.client.get(
                        "/v1/mcp/server/health",
                        headers={"Authorization": f"Bearer {key}"},
                        params=[] if requested is None else [("server_ids", identity) for identity in requested],
                    )
                    assert response.status_code == 200, response.text
                    expected = set(grants) if requested is None else set(grants).intersection(requested)
                    assert {row["server_id"] for row in response.json()}.intersection(owned) == expected, response.text
                    assert all(row["status"] == "healthy" for row in response.json() if row["server_id"] in owned)


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
