from __future__ import annotations

import os
import shutil
import subprocess
import threading
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final

import pytest
from e2e_config import MASTER_KEY, PROXY_BASE_URL
from proxy_client import ProxyClient

pytestmark = pytest.mark.e2e


def _start_cost_map_server(request_log: Path) -> tuple[ThreadingHTTPServer, threading.Thread]:
    class CostMapHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            with request_log.open("a", encoding="utf-8") as log_file:
                log_file.write(f"{self.path}\n")
            body: Final = b'{"test-model": {"litellm_provider": "openai", "mode": "chat"}}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), CostMapHandler)
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _run_lite(
    args: tuple[str, ...],
    server: ThreadingHTTPServer,
    env: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    lite_path: Final = shutil.which("lite")
    assert lite_path is not None, "the installed lite executable is required for e2e coverage"
    try:
        return subprocess.run(
            [lite_path, *args],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
    finally:
        server.shutdown()
        server.server_close()


def _request_count(request_log: Path) -> int:
    return request_log.read_text(encoding="utf-8").count("\n") if request_log.exists() else 0


class TestLiteCliCostMapFetch:
    @pytest.mark.covers("other.cli.model_cost_map.version_skips_fetch")
    def test_lite_version_makes_no_cost_map_request(self, tmp_path: Path) -> None:
        request_log: Final = tmp_path / "requests.log"
        server, thread = _start_cost_map_server(request_log)
        source_root: Final = str(Path(__file__).resolve().parents[3])
        pythonpath: Final = os.pathsep.join(filter(None, (source_root, os.environ.get("PYTHONPATH"))))
        env: Final = {
            **{key: value for key, value in os.environ.items() if key != "LITELLM_LOCAL_MODEL_COST_MAP"},
            "PYTHONPATH": pythonpath,
            "LITELLM_MODEL_COST_MAP_URL": f"http://127.0.0.1:{server.server_port}/map.json",
            "LITELLM_PROXY_URL": PROXY_BASE_URL,
        }
        try:
            result: Final = _run_lite(("--version",), server, env)
        finally:
            thread.join(timeout=10)

        assert result.returncode == 0
        assert "LiteLLM Proxy CLI Version" in result.stdout
        assert _request_count(request_log) == 0

    @pytest.mark.covers("other.cli.model_cost_map.models_list_skips_fetch")
    def test_lite_models_list_uses_proxy_not_cost_map(self, tmp_path: Path, proxy: ProxyClient) -> None:
        model_names: Final = tuple(entry.model_name for entry in proxy.model_info())
        assert model_names
        request_log: Final = tmp_path / "requests.log"
        server, thread = _start_cost_map_server(request_log)
        source_root: Final = str(Path(__file__).resolve().parents[3])
        pythonpath: Final = os.pathsep.join(filter(None, (source_root, os.environ.get("PYTHONPATH"))))
        env: Final = {
            **{key: value for key, value in os.environ.items() if key != "LITELLM_LOCAL_MODEL_COST_MAP"},
            "PYTHONPATH": pythonpath,
            "LITELLM_MODEL_COST_MAP_URL": f"http://127.0.0.1:{server.server_port}/map.json",
            "LITELLM_PROXY_URL": PROXY_BASE_URL,
            "LITELLM_PROXY_API_KEY": MASTER_KEY,
        }
        try:
            result: Final = _run_lite(("models", "list"), server, env)
        finally:
            thread.join(timeout=10)

        assert result.returncode == 0
        assert result.stdout.strip()
        assert any(model_name in result.stdout for model_name in model_names)
        assert _request_count(request_log) == 0
