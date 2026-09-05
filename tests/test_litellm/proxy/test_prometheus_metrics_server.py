"""The separate metrics server must aggregate PROMETHEUS_MULTIPROC_DIR, expose /health and follow its parent's lifetime."""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient
from prometheus_client import values

from litellm.proxy.prometheus_metrics_server import (
    build_metrics_app,
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


def _wait_for_health(port: int) -> httpx.Response:
    deadline: Final = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            return httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
        except httpx.TransportError:
            time.sleep(0.2)
    raise AssertionError(f"metrics server on port {port} never became healthy")


def _wait_until_down(port: int) -> None:
    deadline: Final = time.monotonic() + _SHUTDOWN_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
        except httpx.TransportError:
            return
        time.sleep(0.2)
    raise AssertionError(f"metrics server on port {port} kept running after its parent died")


def test_metrics_app_aggregates_multiproc_dir_and_reports_health(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    _write_worker_sample(pid=1001, value=2)
    _write_worker_sample(pid=1002, value=3)
    other_dir: Final = tmp_path / "other"
    other_dir.mkdir()

    client: Final = TestClient(build_metrics_app(str(tmp_path)))
    metrics: Final = client.get("/metrics")
    assert metrics.status_code == 200
    assert 'litellm_requests_metric_total{model="gpt-5"} 5.0' in metrics.text

    health: Final = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "healthy", "multiproc_dir": str(tmp_path)}

    empty: Final = TestClient(build_metrics_app(str(other_dir))).get("/metrics")
    assert empty.status_code == 200
    assert "litellm_requests_metric_total" not in empty.text


def test_start_metrics_server_process_spawns_module_with_multiproc_dir(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", "/tmp/litellm_multiproc")
    fake_process: Final = MagicMock(pid=4242)
    with (
        patch("subprocess.Popen", return_value=fake_process) as popen,
        patch("atexit.register") as register,
    ):
        assert start_metrics_server_process(host="0.0.0.0", port=4001) is fake_process

    assert tuple(popen.call_args.args[0]) == (
        sys.executable,
        "-m",
        "litellm.proxy.prometheus_metrics_server",
        "--host",
        "0.0.0.0",
        "--port",
        "4001",
        "--multiproc_dir",
        "/tmp/litellm_multiproc",
    )
    register.assert_called_once_with(fake_process.terminate)


def test_start_metrics_server_process_skips_without_multiproc_dir(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    with (
        caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"),
        patch("subprocess.Popen") as popen,
    ):
        assert start_metrics_server_process(host="0.0.0.0", port=4001) is None
    popen.assert_not_called()
    assert "--prometheus_metrics_port ignored" in caplog.text


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
        health: Final = _wait_for_health(port)
        assert health.status_code == 200
        assert health.json()["status"] == "healthy"

        metrics: Final = httpx.get(f"http://127.0.0.1:{port}/metrics", follow_redirects=True, timeout=10.0)
        assert metrics.status_code == 200
        assert 'litellm_requests_metric_total{model="gpt-5"} 7.0' in metrics.text

        parent.kill()
        parent.wait(timeout=10)
        _wait_until_down(port)
    finally:
        parent.kill()
        try:
            os.kill(server_pid, 9)
        except ProcessLookupError:
            pass
