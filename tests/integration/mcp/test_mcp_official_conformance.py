import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway
from integration._support.conformance import authenticated_endpoint, reference_server, run_scenario
from integration._support.mcp import official_client_outcomes, register_mcp
from integration._support.wire import Reply, wire_server
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import ImageContent


@pytest.mark.parametrize("name", ("server-initialize", "tools-list", "tools-call-image"))
def test_official_scenario_through_gateway(gateway: Gateway, tmp_path: Path, unused_tcp_port: int, name: str) -> None:
    root: Final = Path(os.environ["MCP_CONFORMANCE_ROOT"])
    output: Final = Path(os.environ.get("INTEGRATION_RESULTS_DIR", str(tmp_path))) / f"conformance-{uuid.uuid4().hex}"
    with reference_server(root, output, unused_tcp_port) as reference, gateway.scenario() as scenario:
        alias: Final = "official" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, reference, alias, mcp_info={"protocol_version": "2025-11-25"})
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        direct: Final = run_scenario(root, reference.url, name, output / "direct")
        endpoint: Final = str(gateway.client.base_url).rstrip("/") + f"/{alias}/mcp"
        with authenticated_endpoint(endpoint, key, alias) as authenticated:
            proxied: Final = run_scenario(root, authenticated, name, output / "gateway")
        if name == "tools-call-image":

            async def check_image(url: str, tool: str, token: str | None = None) -> None:
                async with httpx.AsyncClient(headers={"Authorization": f"Bearer {token}"} if token else {}) as http:
                    async with streamable_http_client(url, http_client=http) as streams:
                        async with ClientSession(streams[0], streams[1]) as session:
                            await session.initialize()
                            result: Final = await session.call_tool(tool, {})
                            assert result.is_error is False and len(result.content) == 1, result
                            content: Final = result.content[0]
                            assert isinstance(content, ImageContent) and content.mime_type == "image/png", content
                            assert content.data == (
                                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlE"
                                "QVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
                            ), content

            asyncio.run(check_image(reference.url, "test_image_content"))
            asyncio.run(check_image(endpoint, f"{alias}-test_image_content", key))
        assert tuple(check.status for check in direct if check.id == name) == ("SUCCESS",)
        assert tuple(check.status for check in proxied if check.id == name) == ("SUCCESS",)
        listed, called = official_client_outcomes(gateway, key, f"/{alias}/mcp", f"{alias}-test_simple_text", {})
        assert f"{alias}-test_simple_text" in listed.tools, listed
        assert called.ok and called.text == "This is a simple text response for testing.", called


@pytest.mark.parametrize("status", (200, 403))
def test_conformance_bridge_preserves_headers_payload_and_error_status(status: int) -> None:
    body: Final = {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "test_image_content"}}
    response_body: Final = b'data: {"jsonrpc":"2.0","id":7,"error":{"code":-32000,"message":"denied"}}\n\n'
    with wire_server(
        lambda request: Reply(
            status=status, chunks=(response_body[:12], response_body[12:]), content_type="text/event-stream"
        )
    ) as upstream:
        with authenticated_endpoint(upstream.url + "/mcp", "test-key", "official") as endpoint:
            response: Final = httpx.post(
                endpoint,
                json=body,
                headers={
                    "Host": "127.0.0.1",
                    "Origin": "https://untrusted.example",
                    "Authorization": "Bearer replaced",
                    "MCP-Protocol-Version": "2025-03-26",
                    "Mcp-Session-Id": "same-session",
                },
            )
        assert response.status_code == status and response.content == response_body
        requests: Final = upstream.drain()
        assert len(requests) == 1
        received: Final = requests[0]
        assert json.loads(received.body) == {**body, "params": {"name": "official-test_image_content"}}
        assert {
            name: received.headers[name]
            for name in ("host", "origin", "authorization", "mcp-protocol-version", "mcp-session-id")
        } == {
            "host": "127.0.0.1",
            "origin": "https://untrusted.example",
            "authorization": "Bearer test-key",
            "mcp-protocol-version": "2025-03-26",
            "mcp-session-id": "same-session",
        }


def test_official_runner_rejects_unknown_scenario(tmp_path: Path) -> None:
    output: Final = tmp_path / "unknown-scenario"
    with pytest.raises(AssertionError):
        run_scenario(Path(os.environ["MCP_CONFORMANCE_ROOT"]), "http://127.0.0.1:1/mcp", "missing-scenario", output)
    assert "missing-scenario" in (output / "runner.log").read_text()


def test_stalled_reference_is_killed_and_cannot_report_clean_teardown(tmp_path: Path, unused_tcp_port: int) -> None:
    import signal
    from contextlib import ExitStack

    import psutil

    children: Final = frozenset(child.pid for child in psutil.Process().children())
    with ExitStack() as cleanup:
        cleanup.enter_context(reference_server(Path(os.environ["MCP_CONFORMANCE_ROOT"]), tmp_path, unused_tcp_port))
        started: Final = tuple(child for child in psutil.Process().children() if child.pid not in children)
        assert len(started) == 1, started
        victim: Final = started[0]
        victim.send_signal(signal.SIGSTOP)
        with pytest.raises(AssertionError, match="forced cleanup"):
            cleanup.close()
    assert not psutil.pid_exists(victim.pid), "Stopped official reference survived forced cleanup"


def test_reference_children_are_stopped_after_the_root_exits(tmp_path: Path, unused_tcp_port: int) -> None:
    import psutil

    source: Final = tmp_path / "legacy-reference/examples/servers/typescript"
    source.mkdir(parents=True)
    (source / "node_modules").symlink_to(
        Path(os.environ["MCP_CONFORMANCE_ROOT"]) / "legacy-reference/examples/servers/typescript/node_modules",
        target_is_directory=True,
    )
    (source / "everything-server.ts").write_text(
        "import http from 'node:http';\n"
        "import { spawn } from 'node:child_process';\n"
        "import { writeFileSync } from 'node:fs';\n"
        "const child = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], {stdio: 'ignore'});\n"
        "writeFileSync('child.pid', String(child.pid));\n"
        "http.createServer((req, res) => { res.writeHead(400); res.end(); }).listen(Number(process.env.PORT));\n"
        "process.on('SIGTERM', () => process.exit(0));\n"
    )
    with reference_server(tmp_path, tmp_path / "logs", unused_tcp_port):
        child: Final = psutil.Process(int((source / "child.pid").read_text()))
    try:
        assert not child.is_running(), "Reference child survived a clean root exit"
    finally:
        if child.is_running():
            child.kill()
            psutil.wait_procs((child,), timeout=3)
