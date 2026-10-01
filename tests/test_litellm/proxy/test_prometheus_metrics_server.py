"""The separate metrics server must aggregate PROMETHEUS_MULTIPROC_DIR, expose /metrics plus a probe-friendly
/health, and follow its parent's lifetime.

Everything here runs on loopback against a child of this test process; no LLM keys or external network.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from prometheus_client import values

from litellm.proxy.prometheus_metrics_server import (
    PID_HEADER,
    MetricsServerStartupError,
    build_metrics_app,
    main,
    metrics_url,
    start_metrics_server_process,
)

_STARTUP_TIMEOUT_SECONDS: Final = 60.0
_SHUTDOWN_TIMEOUT_SECONDS: Final = 15.0


def _write_worker_sample(pid: int, value: float) -> None:
    """Write one counter sample into PROMETHEUS_MULTIPROC_DIR the way a proxy worker would."""
    counter: Final = values.MultiProcessValue(process_identifier=lambda: pid)(
        "counter",
        "litellm_requests_metric_total",
        "litellm_requests_metric_total",
        ("model",),
        ("gpt-5",),
        "Total number of LLM calls",
    )
    counter.inc(value)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_metrics(port: int, pid: int) -> httpx.Response:
    deadline: Final = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            response: Final = httpx.get(f"http://127.0.0.1:{port}/metrics", follow_redirects=True, timeout=1.0)
            if response.status_code == 200 and response.headers.get(PID_HEADER) == str(pid):
                return response
        except httpx.TransportError:
            pass
        time.sleep(0.2)
    raise AssertionError(f"metrics server on port {port} never served metrics")


def _wait_until_down(port: int) -> None:
    deadline: Final = time.monotonic() + _SHUTDOWN_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            httpx.get(f"http://127.0.0.1:{port}/metrics", timeout=1.0)
        except httpx.TransportError:
            return
        time.sleep(0.2)
    raise AssertionError(f"metrics server on port {port} kept running after its parent died")


def test_metrics_app_aggregates_multiproc_dir_and_reports_pid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    _write_worker_sample(pid=1001, value=2)
    _write_worker_sample(pid=1002, value=3)
    other_dir: Final = tmp_path / "other"
    other_dir.mkdir()

    client: Final = TestClient(build_metrics_app(str(tmp_path)))
    metrics: Final = client.get("/metrics")
    assert metrics.status_code == 200
    assert metrics.headers[PID_HEADER] == str(os.getpid())
    assert 'litellm_requests_metric_total{model="gpt-5"} 5.0' in metrics.text

    health: Final = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "healthy", "multiproc_dir": str(tmp_path)}
    assert client.get("/docs").status_code == 404

    empty: Final = TestClient(build_metrics_app(str(other_dir))).get("/metrics")
    assert empty.status_code == 200
    assert "litellm_requests_metric_total" not in empty.text


@pytest.mark.parametrize(
    ("host", "expected"),
    (
        ("0.0.0.0", "http://127.0.0.1:4001/metrics"),
        ("::", "http://[::1]:4001/metrics"),
        ("10.1.2.3", "http://10.1.2.3:4001/metrics"),
        ("metrics.internal", "http://metrics.internal:4001/metrics"),
    ),
)
def test_metrics_url_probes_loopback_for_wildcard_binds(host: str, expected: str):
    assert metrics_url(host, 4001) == expected


def test_main_serves_the_app_for_the_given_dir_with_uvicorn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    _write_worker_sample(pid=2001, value=6)
    with patch("uvicorn.run") as run:
        main(["--host", "10.1.2.3", "--port", "4001", "--multiproc_dir", str(tmp_path)])

    run.assert_called_once()
    assert run.call_args.kwargs["host"] == "10.1.2.3"
    assert run.call_args.kwargs["port"] == 4001
    client: Final = TestClient(run.call_args.args[0])
    metrics: Final = client.get("/metrics")
    assert metrics.status_code == 200
    assert metrics.headers[PID_HEADER] == str(os.getpid())
    assert 'litellm_requests_metric_total{model="gpt-5"} 6.0' in client.get("/metrics").text


def test_main_falls_back_to_env_multiproc_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    with patch("uvicorn.run") as run:
        main(["--port", "4001"])

    (app,), served_on = run.call_args
    assert served_on["host"] == "0.0.0.0"
    metrics: Final = TestClient(app).get("/metrics")
    assert metrics.status_code == 200
    assert metrics.headers[PID_HEADER] == str(os.getpid())


def test_main_rejects_missing_multiproc_dir(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    with patch("uvicorn.run") as run, pytest.raises(SystemExit) as exit_info:
        main(["--port", "4001"])

    assert exit_info.value.code == 2
    run.assert_not_called()


def test_start_metrics_server_process_returns_only_once_child_serves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    _write_worker_sample(pid=3001, value=4)
    port: Final = _free_port()
    with patch("atexit.register") as register:
        process: Final = start_metrics_server_process(host="127.0.0.1", port=port, multiproc_dir=str(tmp_path))
    try:
        register.assert_called_once_with(process.terminate)
        assert process.poll() is None
        startup_metrics: Final = httpx.get(f"http://127.0.0.1:{port}/metrics", follow_redirects=True, timeout=5.0)
        assert startup_metrics.status_code == 200
        assert startup_metrics.headers[PID_HEADER] == str(process.pid)
        metrics: Final = httpx.get(f"http://127.0.0.1:{port}/metrics", follow_redirects=True, timeout=10.0)
        assert 'litellm_requests_metric_total{model="gpt-5"} 4.0' in metrics.text
    finally:
        process.kill()
        process.wait(timeout=10)


def test_start_metrics_server_process_fails_when_port_is_taken(tmp_path: Path):
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port: Final = occupied.getsockname()[1]
        with (
            patch("atexit.register"),
            pytest.raises(
                MetricsServerStartupError, match=rf"exited with code [1-9]\d* before serving 127.0.0.1:{port}"
            ),
        ):
            start_metrics_server_process(host="127.0.0.1", port=port, multiproc_dir=str(tmp_path))


class _ImpostorMetrics(BaseHTTPRequestHandler):
    """An unrelated service already on the port that answers /metrics with 200 and plausible metrics."""

    def do_GET(self) -> None:
        body: Final = b"# HELP impostor_metric A plausible metric\n# TYPE impostor_metric counter\nimpostor_metric 1\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


def test_start_metrics_server_process_rejects_metrics_from_another_service_on_the_port(tmp_path: Path):
    with ThreadingHTTPServer(("127.0.0.1", 0), _ImpostorMetrics) as impostor:
        threading.Thread(target=impostor.serve_forever, daemon=True).start()
        port: Final = impostor.server_address[1]
        impostor_response: Final = httpx.get(f"http://127.0.0.1:{port}/metrics")
        assert impostor_response.status_code == 200
        assert "# HELP impostor_metric" in impostor_response.text
        assert PID_HEADER not in impostor_response.headers
        with (
            patch("atexit.register"),
            pytest.raises(MetricsServerStartupError, match=rf"exited with code [1-9]\d* before serving 127.0.0.1:{port}"),
        ):
            start_metrics_server_process(host="127.0.0.1", port=port, multiproc_dir=str(tmp_path))
        impostor.shutdown()


def test_metrics_server_process_serves_and_exits_with_parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    _write_worker_sample(pid=2001, value=7)
    port: Final = _free_port()
    server_argv: Final = (
        sys.executable,
        "-m",
        "litellm.proxy.prometheus_metrics_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--multiproc_dir",
        str(tmp_path),
    )
    parent: Final = subprocess.Popen(
        (
            sys.executable,
            "-c",
            "import subprocess, sys, time; p = subprocess.Popen(sys.argv[1:]); print(p.pid, flush=True); time.sleep(600)",
            *server_argv,
        ),
        stdout=subprocess.PIPE,
        text=True,
    )
    assert parent.stdout is not None
    server_pid: Final = int(parent.stdout.readline())
    try:
        metrics: Final = _wait_for_metrics(port, server_pid)
        assert metrics.status_code == 200
        assert metrics.headers[PID_HEADER] == str(server_pid)

        scrape: Final = httpx.get(f"http://127.0.0.1:{port}/metrics", follow_redirects=True, timeout=10.0)
        assert scrape.status_code == 200
        assert scrape.headers[PID_HEADER] == str(server_pid)
        assert 'litellm_requests_metric_total{model="gpt-5"} 7.0' in scrape.text

        parent.kill()
        parent.wait(timeout=10)
        _wait_until_down(port)
    finally:
        parent.kill()
        try:
            os.kill(server_pid, 9)
        except ProcessLookupError:
            pass
