import uuid
from contextlib import ExitStack
from typing import Final

import pytest
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule, run_state_machine_as_test

from integration._support.client import Gateway
from integration._support.database import read_rows
from integration._support.generation import LIFECYCLE_SETTINGS, bounded_http_requests
from integration._support.mcp import call_tool, mcp_peer, register_mcp, tool_names


@pytest.mark.covers("mcp.call_tool.saved_headers.reach_actual_transport")
def test_saved_headers_reach_real_mcp_tool_and_survive_unrelated_edit(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "integration" + uuid.uuid4().hex
        identity: Final = register_mcp(scenario, peer, alias, static_headers={"X-Integration-Saved": "synthetic-header-value"})
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
                updated: Final = gateway.request("PUT", "/v1/mcp/server", {"server_id": identity, "server_name": alias + "renamed"})
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
                    self.identity = register_mcp(scenario, peer, self.name, static_headers={"X-Integration-Saved": self.marker})
                    self.key = scenario.key(object_permission={"mcp_servers": [self.identity]})
                except BaseException:
                    with budget.cleanup():
                        self.resources.close()
                    raise

            @rule(value=st.sampled_from(("first", "second", "third")))
            def header(self, value: str) -> None:
                response: Final = gateway.request("PUT", "/v1/mcp/server", {"server_id": self.identity, "static_headers": {"X-Integration-Saved": value}})
                assert response.status_code == 202, response.text
                self.marker = value

            @rule(value=st.sampled_from(("original", "renamed")))
            def rename(self, value: str) -> None:
                response: Final = gateway.request("PUT", "/v1/mcp/server", {"server_id": self.identity, "server_name": self.name + value})
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
                assert len(read_rows('SELECT server_id FROM "LiteLLM_MCPServerTable" WHERE server_id=%s', (self.identity,))) == 1

            def teardown(self) -> None:
                with budget.cleanup():
                    self.resources.close()

        run_state_machine_as_test(Servers, settings=LIFECYCLE_SETTINGS)
