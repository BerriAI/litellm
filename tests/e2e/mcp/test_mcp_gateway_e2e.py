"""Live e2e: MCP gateway tool access and the per-server concurrency cap.

Covers mcp.list_tools.api_key.succeeds + mcp.call_tool.api_key.succeeds (the
`x-litellm-api-key` header), mcp.list_tools.bearer.succeeds +
mcp.call_tool.bearer.succeeds (the `Authorization` header), and
mcp.call_tool.api_key.caps_concurrency (`max_concurrent_requests`), all
against the deterministic mcp-stub compose service (tests/e2e/mcp/stub/).

Each test walks the full lifecycle: register the server over the management
API and defer its deletion, assert the recorded state round-trips
(GET /v1/mcp/server/{id} echoes what was configured), then drive the real MCP
protocol through the gateway the way a production MCP host does (initialize,
tools/list, tools/call over streamable HTTP at {PROXY}/{alias}/mcp) and
assert the enforced behavior. Both auth headers carry `Bearer <key>`; that is
the documented contract on the MCP routes (a bare key in `x-litellm-api-key`
is accepted on LLM routes but 401s here).

The concurrency test's observable is the stub's own per-marker in-flight
counter, read back through the proxy via the stub's `stats` tool. The
proxy-side semaphore queues excess tool calls instead of rejecting them, so
under a cap every call in a simultaneous burst must still succeed while the
stub never sees more than the cap overlap. The uncapped control server drives
the identical burst through the identical machinery and must see the whole
burst overlap; that is the fail-before-fix evidence built into the test,
since a broken (or skipped) semaphore makes the capped server record the full
burst exactly like the control and the equality assertion fail.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from e2e_config import MCP_STUB_URL, unique_marker
from lifecycle import ResourceManager
from mcp_client import McpAuth, McpClient, McpDenied, McpToolText
from models import McpServerCreateBody

pytestmark = pytest.mark.e2e

STUB_TOOLS = ("echo", "slow_echo", "stats")


class TestMcpToolAccess:
    """Any virtual key reaches an allow_all_keys server through either
    documented auth header; a key the proxy does not recognize is turned away
    at the door."""

    @pytest.mark.covers("mcp.list_tools.api_key.succeeds")
    @pytest.mark.covers("mcp.call_tool.api_key.succeeds")
    def test_list_and_call_tools_with_x_litellm_api_key_header(
        self, client: McpClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        alias = f"e2emcp{unique_marker()}"
        created = client.create_server(McpServerCreateBody(alias=alias, url=MCP_STUB_URL, allow_all_keys=True))
        resources.defer(lambda: client.delete_server(created.server_id))

        stored = client.server_info(created.server_id)
        assert stored.alias == alias
        assert stored.url == MCP_STUB_URL
        assert stored.allow_all_keys is True

        auth = McpAuth(header_name="x-litellm-api-key", key=scoped_key)
        names = client.poll_tool_names(alias, auth)
        expected = tuple(sorted(f"{alias}-{tool}" for tool in STUB_TOOLS))
        assert names == expected, f"gateway listed {names}, expected exactly {expected}"

        payload = f"e2e-{unique_marker()}"
        result = client.call_tool(alias, auth, f"{alias}-echo", {"text": payload})
        assert result.is_error is False, f"echo call errored: {result.text[:300]}"
        assert result.text == payload

        denied = client.list_tools_once(alias, McpAuth(header_name="x-litellm-api-key", key="sk-not-a-real-key"))
        assert isinstance(denied, McpDenied), f"unrecognized key was served tools: {denied}"
        assert denied.status_code == 401, f"expected 401 for an unrecognized key, got {denied}"

    @pytest.mark.covers("mcp.list_tools.bearer.succeeds")
    @pytest.mark.covers("mcp.call_tool.bearer.succeeds")
    def test_list_and_call_tools_with_authorization_bearer_header(
        self, client: McpClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        alias = f"e2emcp{unique_marker()}"
        created = client.create_server(McpServerCreateBody(alias=alias, url=MCP_STUB_URL, allow_all_keys=True))
        resources.defer(lambda: client.delete_server(created.server_id))

        stored = client.server_info(created.server_id)
        assert stored.alias == alias
        assert stored.url == MCP_STUB_URL

        auth = McpAuth(header_name="Authorization", key=scoped_key)
        names = client.poll_tool_names(alias, auth)
        expected = tuple(sorted(f"{alias}-{tool}" for tool in STUB_TOOLS))
        assert names == expected, f"gateway listed {names}, expected exactly {expected}"

        payload = f"e2e-{unique_marker()}"
        result = client.call_tool(alias, auth, f"{alias}-echo", {"text": payload})
        assert result.is_error is False, f"echo call errored: {result.text[:300]}"
        assert result.text == payload


class TestMcpServerMaxConcurrency:
    """`max_concurrent_requests` bounds how many tool calls the gateway lets
    reach one server at a time; excess calls queue and still succeed."""

    @pytest.mark.covers("mcp.call_tool.api_key.caps_concurrency")
    def test_max_concurrent_requests_caps_in_flight_upstream_calls(
        self, client: McpClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        """Two servers against the same stub: one capped at 2 concurrent
        requests, one uncapped control. The tools/list polls are the settle
        step; they hold until the just-created records are servable on the
        gateway (and the fresh key has cleared the auth cache) so the bursts
        measure the semaphore, not propagation. Each burst is 6 simultaneous
        slow_echo calls, one per thread, each over its own MCP session."""
        max_concurrent = 2
        burst_size = 6
        slow_call_seconds = 2.0
        auth = McpAuth(header_name="x-litellm-api-key", key=scoped_key)

        capped_alias = f"e2emcpcap{unique_marker()}"
        capped = client.create_server(
            McpServerCreateBody(
                alias=capped_alias,
                url=MCP_STUB_URL,
                allow_all_keys=True,
                max_concurrent_requests=max_concurrent,
            )
        )
        resources.defer(lambda: client.delete_server(capped.server_id))

        control_alias = f"e2emcpfree{unique_marker()}"
        control = client.create_server(McpServerCreateBody(alias=control_alias, url=MCP_STUB_URL, allow_all_keys=True))
        resources.defer(lambda: client.delete_server(control.server_id))

        assert client.server_info(capped.server_id).max_concurrent_requests == max_concurrent
        assert client.server_info(control.server_id).max_concurrent_requests is None

        _ = client.poll_tool_names(capped_alias, auth)
        _ = client.poll_tool_names(control_alias, auth)

        def burst_of_slow_calls(alias: str, text: str, marker: str) -> list[McpToolText]:
            arguments = {"text": text, "marker": marker, "sleep_seconds": slow_call_seconds}
            with ThreadPoolExecutor(max_workers=burst_size) as pool:
                futures = [
                    pool.submit(client.call_tool, alias, auth, f"{alias}-slow_echo", arguments)
                    for _ in range(burst_size)
                ]
                return [future.result() for future in futures]

        capped_marker = unique_marker()
        capped_results = burst_of_slow_calls(capped_alias, "capped", capped_marker)
        assert len(capped_results) == burst_size
        assert all(result.is_error is False and result.text == "capped" for result in capped_results), (
            f"queued calls must all still succeed under the cap: {capped_results}"
        )

        capped_stats = client.stub_stats(capped_alias, auth, f"{capped_alias}-stats", capped_marker)
        assert capped_stats.completed == burst_size
        assert capped_stats.max_in_flight == max_concurrent, (
            f"stub saw {capped_stats.max_in_flight} overlapping calls; the cap of {max_concurrent} "
            f"must be the exact ceiling AND be reached ({burst_size} queued calls saturate it)"
        )

        control_marker = unique_marker()
        control_results = burst_of_slow_calls(control_alias, "control", control_marker)
        assert all(result.is_error is False and result.text == "control" for result in control_results)

        control_stats = client.stub_stats(control_alias, auth, f"{control_alias}-stats", control_marker)
        assert control_stats.completed == burst_size
        assert control_stats.max_in_flight == burst_size, (
            f"uncapped control saw {control_stats.max_in_flight} overlapping calls, expected all {burst_size}; "
            f"if this fails the instrument cannot detect over-cap concurrency and the capped assertion is vacuous"
        )
