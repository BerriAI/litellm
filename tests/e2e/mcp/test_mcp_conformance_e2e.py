"""Live e2e: the official MCP conformance suite, run through the gateway.

Covers mcp.protocol.api_key.passes_official_conformance: the gateway is a
protocol middlebox, so beyond this suite's behavior tests it must stay
conformant to the MCP spec as re-served to hosts. This test drives the
official @modelcontextprotocol/conformance server scenarios (the same suite
the SDKs gate their CI on) against a gateway-registered server backed by the
stub's /conformance mount, which implements the suite's hardcoded fixture
contract (test_simple_text, test://static-text, test_simple_prompt, ...).

The conformance CLI cannot send custom headers, so the run goes through
auth_forwarder.serve(), an in-process reverse proxy that stamps the virtual
key onto every request the way a configured MCP host's HTTP layer would; the
gateway path itself stays fully authenticated and unmodified.

EXPECTED_GAPS is the checked-in baseline, exact in both directions: a
scenario failing that is not listed fails this test (a regression), and a
listed scenario that starts passing also fails it (so the baseline ratchets
down instead of going stale). The current entries are real gateway findings,
verified by hand: logging/setLevel and completion/complete are not relayed
(-32601), prompts/get resolves alias-prefixed names only (the unprefixed
fallback that tools/call has is missing, so the suite's hardcoded prompt
names 403), and binary resource read-through drops the base64 blob field
(the stub serves it; the gateway's answer has no blob).

Scenarios needing client capabilities the gateway does not advertise for its
callers (sampling, elicitation) and SSE-timing scenarios are intentionally
not run; the curated list below is the subset a gateway can honestly own.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

import auth_forwarder
from e2e_config import MCP_STUB_CONFORMANCE_URL, unique_marker
from mcp_client import McpClient
from models import KeyGenerateBody, McpServerCreateBody

pytestmark = pytest.mark.e2e

CONFORMANCE_PACKAGE = "@modelcontextprotocol/conformance@0.1.11"

SCENARIOS = (
    "server-initialize",
    "ping",
    "logging-set-level",
    "completion-complete",
    "tools-list",
    "tools-call-simple-text",
    "tools-call-error",
    "resources-list",
    "resources-read-text",
    "resources-read-binary",
    "resources-templates-read",
    "prompts-list",
    "prompts-get-simple",
    "prompts-get-with-args",
)

EXPECTED_GAPS = {
    "logging-set-level": "gateway answers logging/setLevel with -32601 Method not found instead of relaying it",
    "completion-complete": "gateway answers completion/complete with -32601 Method not found instead of relaying it",
    "prompts-get-simple": "prompts/get lacks the unprefixed-name fallback tools/call has; unprefixed names 403",
    "prompts-get-with-args": "prompts/get lacks the unprefixed-name fallback tools/call has; unprefixed names 403",
    "resources-read-binary": "gateway drops the base64 blob field on binary resource read-through",
}


@pytest.fixture(scope="module")
def conformance_target(client: McpClient) -> Iterator[str]:
    """A conformance-fixture server registered on the gateway, fronted by the
    key-stamping forwarder; yields the URL the conformance CLI tests. Module
    scoped so all scenarios share one server, one key, and one forwarder."""
    if shutil.which("npx") is None:
        pytest.skip("npx not available; the official conformance suite runs on Node")
    alias = f"e2emcpconf{unique_marker()}"
    created = client.create_server(
        McpServerCreateBody(alias=alias, url=MCP_STUB_CONFORMANCE_URL, allow_all_keys=True)
    )
    key = client.gateway.generate_key(KeyGenerateBody())
    try:
        _ = client.poll_tool_names(alias, {"x-litellm-api-key": f"Bearer {key}"})
        with auth_forwarder.serve(key) as forwarder_base:
            yield f"{forwarder_base}/{alias}/mcp"
    finally:
        client.delete_server(created.server_id)
        client.gateway.delete_key(key)


class TestMcpOfficialConformance:
    """Each curated official scenario passes through the gateway, except the
    exact set of known gaps pinned in EXPECTED_GAPS."""

    @pytest.mark.covers("mcp.protocol.api_key.passes_official_conformance")
    @pytest.mark.parametrize("scenario", SCENARIOS)
    def test_official_scenario(self, scenario: str, conformance_target: str, tmp_path: Path) -> None:
        run = subprocess.run(
            ["npx", "-y", CONFORMANCE_PACKAGE, "server", "--url", conformance_target, "--scenario", scenario],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=tmp_path,
        )
        failed = run.returncode != 0
        expected_reason = EXPECTED_GAPS.get(scenario)
        if expected_reason is not None:
            assert failed, (
                f"conformance scenario {scenario!r} now PASSES; the gateway gap "
                f"({expected_reason}) appears fixed, so remove it from EXPECTED_GAPS to ratchet the baseline"
            )
            return
        assert not failed, (
            f"conformance scenario {scenario!r} failed through the gateway (exit {run.returncode}); "
            f"output tail:\n{run.stdout[-1500:]}"
        )
