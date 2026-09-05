"""Serve Prometheus `/metrics` from its own process so a scrape never runs on an inference worker.

Workers write their samples to `PROMETHEUS_MULTIPROC_DIR`; this process reads them back with a
``MultiProcessCollector`` and serves the aggregated output on a separate port. The proxy CLI starts
it with ``--prometheus_metrics_port``. It can also run as a sidecar sharing the same directory:
``python -m litellm.proxy.prometheus_metrics_server --host 0.0.0.0 --port 4001``.
"""

from __future__ import annotations

import argparse
import atexit
import os
import subprocess
import sys
import threading
import time
from collections.abc import Sequence
from types import MappingProxyType
from typing import Final

import httpx
from fastapi import FastAPI
from prometheus_client import CollectorRegistry, multiprocess
from pydantic import BaseModel

from litellm.integrations.prometheus_metrics_endpoint import make_metrics_asgi_app

HEALTH_PATH: Final = "/health"
_PARENT_POLL_INTERVAL_SECONDS: Final = 1.0
_STARTUP_TIMEOUT_SECONDS: Final = 30.0
_STARTUP_POLL_INTERVAL_SECONDS: Final = 0.1
_WILDCARD_TO_LOOPBACK: Final = MappingProxyType({"0.0.0.0": "127.0.0.1", "::": "::1"})


class MetricsServerStartupError(RuntimeError):
    """The metrics process died or never answered its health check before the proxy started serving."""


class MetricsServerHealth(BaseModel):
    status: str
    multiproc_dir: str


def build_metrics_app(multiproc_dir: str) -> FastAPI:
    registry: Final = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry, path=multiproc_dir)
    app: Final = FastAPI(title="LiteLLM Prometheus metrics", docs_url=None, redoc_url=None, openapi_url=None)
    app.mount("/metrics", make_metrics_asgi_app(registry))

    @app.get(HEALTH_PATH)
    def health() -> MetricsServerHealth:
        return MetricsServerHealth(status="healthy", multiproc_dir=multiproc_dir)

    return app


def _exit_when_parent_dies(parent_pid: int) -> None:
    def watch() -> None:
        while os.getppid() == parent_pid:
            time.sleep(_PARENT_POLL_INTERVAL_SECONDS)
        os._exit(0)

    threading.Thread(target=watch, name="litellm-metrics-parent-watchdog", daemon=True).start()


def run_metrics_server(host: str, port: int, multiproc_dir: str) -> None:
    import uvicorn

    _exit_when_parent_dies(os.getppid())
    uvicorn.run(build_metrics_app(multiproc_dir), host=host, port=port, log_level="warning", access_log=False)


def health_url(host: str, port: int) -> str:
    probe_host: Final = _WILDCARD_TO_LOOPBACK.get(host, host)
    netloc: Final = f"[{probe_host}]" if ":" in probe_host else probe_host
    return f"http://{netloc}:{port}{HEALTH_PATH}"


def _wait_until_serving(process: subprocess.Popen[bytes], host: str, port: int) -> None:
    url: Final = health_url(host, port)
    deadline: Final = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if (returncode := process.poll()) is not None:
            raise MetricsServerStartupError(
                f"Prometheus metrics server exited with code {returncode} before serving {host}:{port}; "
                "is the port already in use?"
            )
        try:
            if httpx.get(url, timeout=1.0).status_code == 200:
                return
        except httpx.TransportError:
            time.sleep(_STARTUP_POLL_INTERVAL_SECONDS)
    process.terminate()
    raise MetricsServerStartupError(
        f"Prometheus metrics server did not answer {url} within {_STARTUP_TIMEOUT_SECONDS:.0f}s"
    )


def start_metrics_server_process(host: str, port: int, multiproc_dir: str) -> subprocess.Popen[bytes]:
    """Spawn the metrics server next to the proxy and block until it answers its health check."""
    process: Final = subprocess.Popen(
        (
            sys.executable,
            "-m",
            "litellm.proxy.prometheus_metrics_server",
            "--host",
            host,
            "--port",
            str(port),
            "--multiproc_dir",
            multiproc_dir,
        )
    )
    atexit.register(process.terminate)
    _wait_until_serving(process, host, port)
    return process


def main(argv: Sequence[str] | None = None) -> None:
    parser: Final = argparse.ArgumentParser(
        description="Serve LiteLLM Prometheus metrics from PROMETHEUS_MULTIPROC_DIR"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--multiproc_dir", default=os.environ.get("PROMETHEUS_MULTIPROC_DIR"))
    args: Final = parser.parse_args(argv)
    if not args.multiproc_dir:
        parser.error("--multiproc_dir or PROMETHEUS_MULTIPROC_DIR is required")
    run_metrics_server(host=args.host, port=args.port, multiproc_dir=args.multiproc_dir)


if __name__ == "__main__":
    main()
