"""Golden-path smoke tests for the MCP docs examples (see docs_snippets/README.md).

Executes the committed doc snippets against a live in-process proxy backed by a
local mock MCP server, so a failure here means a documented MCP example drifted
from real gateway behavior.
"""

import asyncio
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import typing
from pathlib import Path
from typing import Final

import httpx
import pytest
import uvicorn
import yaml
from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client

from litellm.proxy.proxy_server import (
    app as proxy_app,
)
from litellm.proxy.proxy_server import (
    cleanup_router_config_variables,
    initialize,
)

SNIPPETS_DIR: Final = Path(__file__).parent / "docs_snippets"
PLACES_SERVER_SCRIPT: Final = Path(__file__).parent / "places_mcp_server.py"
PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
START_TIMEOUT: Final = 30
MASTER_KEY: Final = "sk-1234"

SHELL_SNIPPETS: Final = tuple(sorted(SNIPPETS_DIR.glob("*.sh")))
PROVIDER_BACKED_SNIPPETS: Final = frozenset({"responses_embedded.sh", "chat_completions_mcp.sh"})
SKIP_NO_OPENAI_KEY: Final = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="provider-backed docs snippet: set OPENAI_API_KEY to run it",
)


def _snippet_source(path: Path) -> str:
    first_line: Final = path.read_text().splitlines()[0]
    assert first_line.startswith("# source:"), f"{path} must start with a '# source:' line naming the doc it mirrors"
    return first_line.removeprefix("# source:").strip()


def _fail_context(path: Path) -> str:
    return f"snippet {path.relative_to(PROJECT_ROOT)} (from {_snippet_source(path)})"


def _run_snippet(path: Path, base_url: str) -> str:
    rendered: Final = (
        path.read_text()
        .replace("http://localhost:4000", base_url)
        .replace("https://your-proxy.com", base_url)
        .replace("<your-litellm-proxy-base-url>", base_url)
    )
    env: Final = {**os.environ, "LITELLM_API_KEY": MASTER_KEY}
    completed: Final = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", rendered],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert completed.returncode == 0, (
        f"{_fail_context(path)} exited {completed.returncode}\n"
        f"command:\n{rendered}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return completed.stdout


def _run_json_snippet(path: Path, base_url: str) -> typing.Any:
    stdout: Final = _run_snippet(path, base_url)
    try:
        return json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{_fail_context(path)} did not return JSON: {exc}\noutput:\n{stdout}") from exc


class TestSnippetsParse:
    @pytest.mark.parametrize("path", SHELL_SNIPPETS, ids=lambda p: p.name)
    def test_shell_snippet_is_valid_bash(self, path: Path) -> None:
        completed: Final = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        assert completed.returncode == 0, f"{_fail_context(path)} is not valid bash:\n{completed.stderr}"

    @pytest.mark.parametrize("path", SHELL_SNIPPETS, ids=lambda p: p.name)
    def test_shell_snippet_json_body_parses(self, path: Path) -> None:
        match: Final = re.search(r"(?:--data|-d)\s+'(.*?)'", path.read_text(), re.DOTALL)
        if match is None:
            pytest.skip(f"{path.name} sends no JSON body")
        try:
            json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise AssertionError(f"{_fail_context(path)} has an invalid JSON body: {exc}") from exc

    def test_config_snippet_parses_and_keeps_golden_fields(self) -> None:
        config_path: Final = SNIPPETS_DIR / "config.yaml"
        config: Final = yaml.safe_load(config_path.read_text())
        places: Final = config["mcp_servers"]["places_api"]
        assert places["transport"] == "http", _fail_context(config_path)
        assert places["allow_all_keys"] is True, _fail_context(config_path)
        assert config["general_settings"]["master_key"] == MASTER_KEY, _fail_context(config_path)


@pytest.fixture(scope="session", autouse=True)
def _proxy_session_env() -> typing.Iterator[None]:
    mp: Final = pytest.MonkeyPatch()
    mp.delenv("DATABASE_URL", raising=False)
    mp.setenv("LITELLM_MASTER_KEY", MASTER_KEY)
    try:
        yield
    finally:
        mp.undo()


@pytest.fixture(scope="session")
def places_mcp_server_url() -> typing.Iterator[str]:
    host: Final = "127.0.0.1"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        _, port = sock.getsockname()

    server_process: Final = subprocess.Popen(
        [sys.executable, str(PLACES_SERVER_SCRIPT), "--host", host, "--port", str(port)],
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    start_time: Final = time.time()
    while True:
        if server_process.poll() is not None:
            stdout, stderr = server_process.communicate()
            raise RuntimeError(f"places MCP server exited early.\nSTDOUT: {stdout.decode()}\nSTDERR: {stderr.decode()}")
        try:
            with socket.create_connection((host, port), timeout=0.1):
                break
        except OSError:
            if time.time() - start_time > START_TIMEOUT:
                server_process.terminate()
                raise TimeoutError("places MCP server did not start in time")
            time.sleep(0.05)

    yield f"http://{host}:{port}/mcp"

    server_process.terminate()
    try:
        server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server_process.kill()


@pytest.fixture(scope="session")
def proxy_base_url(tmp_path_factory: pytest.TempPathFactory, places_mcp_server_url: str) -> typing.Iterator[str]:
    config_path: Final = tmp_path_factory.mktemp("mcp_docs_smoke") / "config.yaml"
    config_path.write_text(
        (SNIPPETS_DIR / "config.yaml").read_text().replace("{{PLACES_MCP_URL}}", places_mcp_server_url)
    )

    cleanup_router_config_variables()
    asyncio.run(initialize(config=str(config_path), debug=True))

    sock: Final = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    host, port = sock.getsockname()
    server: Final = uvicorn.Server(uvicorn.Config(proxy_app, host=host, port=port, log_level="warning"))

    def _run() -> None:
        loop: Final = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve(sockets=[sock]))

    thread: Final = threading.Thread(target=_run, daemon=True)
    thread.start()
    start_time: Final = time.time()
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("Proxy server failed to start")
        if time.time() - start_time > START_TIMEOUT:
            raise TimeoutError("Proxy server did not start in time")
        time.sleep(0.05)

    yield f"http://{host}:{port}"

    server.should_exit = True
    thread.join(timeout=10)
    sock.close()


class TestDocsGoldenPath:
    def test_gateway_health(self, proxy_base_url: str) -> None:
        response: Final = httpx.get(f"{proxy_base_url}/health/liveliness")
        assert response.status_code == 200, response.text

    def test_rest_list_servers(self, proxy_base_url: str) -> None:
        path: Final = SNIPPETS_DIR / "rest_list_servers.sh"
        servers: Final = _run_json_snippet(path, proxy_base_url)
        server_names: Final = {server.get("server_name") for server in servers}
        assert "places_api" in server_names, (
            f"{_fail_context(path)}: expected server_name 'places_api' in {server_names}"
        )

    def test_rest_tools_list_all_servers(self, proxy_base_url: str) -> None:
        path: Final = SNIPPETS_DIR / "rest_tools_list_all.sh"
        payload: Final = _run_json_snippet(path, proxy_base_url)
        tools: Final = payload["tools"] if isinstance(payload, dict) else payload
        by_server: Final = {tool["name"]: tool.get("mcp_info", {}).get("server_name") for tool in tools}
        assert "getPlaces" in by_server, f"{_fail_context(path)}: expected tool 'getPlaces' in {sorted(by_server)}"
        assert by_server["getPlaces"] == "places_api", _fail_context(path)

    def test_rest_tools_list_one_server(self, proxy_base_url: str) -> None:
        path: Final = SNIPPETS_DIR / "rest_tools_list_one_server.sh"
        payload: Final = _run_json_snippet(path, proxy_base_url)
        tools: Final = payload["tools"] if isinstance(payload, dict) else payload
        tool_names: Final = {tool["name"] for tool in tools}
        assert "getPlaces" in tool_names, f"{_fail_context(path)}: expected unprefixed tool 'getPlaces' in {tool_names}"

    @pytest.mark.parametrize(
        "snippet_name",
        ["rest_tools_call_unprefixed.sh", "rest_tools_call_prefixed.sh"],
    )
    def test_rest_tools_call(self, proxy_base_url: str, snippet_name: str) -> None:
        _run_json_snippet(SNIPPETS_DIR / "rest_tools_list_all.sh", proxy_base_url)
        path: Final = SNIPPETS_DIR / snippet_name
        result: Final = _run_json_snippet(path, proxy_base_url)
        assert result.get("isError") in (None, False), f"{_fail_context(path)}: tool call errored: {result}"
        text: Final = json.dumps(result)
        assert "Blue Bottle Coffee (coffee)" in text, f"{_fail_context(path)}: expected mock tool output in {text}"

    @pytest.mark.asyncio
    async def test_mcp_protocol_discovery_and_call(self, proxy_base_url: str) -> None:
        async with asyncio.timeout(30):
            async with streamablehttp_client(
                url=f"{proxy_base_url}/mcp",
                headers={
                    "x-litellm-api-key": f"Bearer {MASTER_KEY}",
                    "x-mcp-servers": "places_api",
                },
            ) as (read, write, _get_session_id):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_result: Final = await session.list_tools()
                    tool_names: Final = {tool.name for tool in tools_result.tools}
                    assert any(name.endswith("getPlaces") for name in tool_names), (
                        f"expected a getPlaces tool over /mcp, got {tool_names}"
                    )
                    result: Final = await session.call_tool("getPlaces", arguments={"query": "coffee"})
                    first_content: Final = result.content[0]
                    assert isinstance(first_content, types.TextContent), (
                        f"expected TextContent from getPlaces, got {type(first_content)}"
                    )
                    assert "Blue Bottle Coffee (coffee)" in first_content.text


class TestDocsProviderBackedPath:
    @SKIP_NO_OPENAI_KEY
    def test_responses_embedded_mcp(self, proxy_base_url: str) -> None:
        path: Final = SNIPPETS_DIR / "responses_embedded.sh"
        response: Final = _run_json_snippet(path, proxy_base_url)
        assert response.get("status") == "completed", f"{_fail_context(path)}: {response}"
        output_types: Final = [item.get("type") for item in response.get("output", [])]
        assert "mcp_tools_fetched" in output_types, (
            f"{_fail_context(path)}: expected mcp_tools_fetched in {output_types}"
        )
        assert "tool_execution_results" in output_types, (
            f"{_fail_context(path)}: expected tool_execution_results in {output_types}"
        )

    @SKIP_NO_OPENAI_KEY
    def test_chat_completions_embedded_mcp(self, proxy_base_url: str) -> None:
        path: Final = SNIPPETS_DIR / "chat_completions_mcp.sh"
        response: Final = _run_json_snippet(path, proxy_base_url)
        content: Final = response["choices"][0]["message"]["content"]
        assert content, f"{_fail_context(path)}: empty assistant reply: {response}"
        assert "Blue Bottle" in content, f"{_fail_context(path)}: expected tool result to reach the reply: {content}"
