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
from contextlib import closing
from types import MappingProxyType
from typing import Final

import httpx
from fastapi import FastAPI
from prometheus_client import CollectorRegistry, multiprocess
from pydantic import BaseModel, ConfigDict
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from litellm.integrations.prometheus_metrics_endpoint import make_metrics_asgi_app
from litellm.llms.custom_httpx.http_handler import HTTPHandler

METRICS_PATH: Final = "/metrics"
PID_HEADER: Final = "x-litellm-metrics-pid"
_PARENT_POLL_INTERVAL_SECONDS: Final = 1.0
_STARTUP_TIMEOUT_SECONDS: Final = 30.0
_STARTUP_POLL_INTERVAL_SECONDS: Final = 0.1
_STARTUP_PROBE_TIMEOUT_SECONDS: Final = 1.0
_WILDCARD_TO_LOOPBACK: Final = MappingProxyType({"0.0.0.0": "127.0.0.1", "::": "::1"})


class _CliArgs(BaseModel):
    model_config = ConfigDict(frozen=True)

    host: str
    port: int
    multiproc_dir: str | None


class MetricsServerStartupError(RuntimeError):
    """The metrics process died or never answered on its port before the proxy started serving."""


def _add_pid_header(app: ASGIApp) -> ASGIApp:
    async def app_with_pid(scope: Scope, receive: Receive, send: Send) -> None:
        async def send_with_pid(message: Message) -> None:
            if message["type"] == "http.response.start":
                await send(
                    {
                        **message,
                        "headers": [
                            *message["headers"],
                            (PID_HEADER.encode(), str(os.getpid()).encode()),
                        ],
                    }
                )
                return
            await send(message)

        await app(scope, receive, send_with_pid)

    return app_with_pid


def build_metrics_app(multiproc_dir: str) -> FastAPI:
    registry: Final = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry, path=multiproc_dir)
    app: Final = FastAPI(title="LiteLLM Prometheus metrics", docs_url=None, redoc_url=None, openapi_url=None)
    app.mount(METRICS_PATH, _add_pid_header(make_metrics_asgi_app(registry)))

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


def metrics_url(host: str, port: int) -> str:
    probe_host: Final = _WILDCARD_TO_LOOPBACK.get(host, host)
    netloc: Final = f"[{probe_host}]" if ":" in probe_host else probe_host
    return f"http://{netloc}:{port}{METRICS_PATH}"


def _answered_by(http: HTTPHandler, url: str, pid: int) -> bool:
    """True only when the metrics response comes from our child, not from whatever else holds the port."""
    try:
        response: Final = http.get(url)  # pyright: ignore[reportUnknownMemberType]  # HTTPHandler.get exposes untyped optional mappings
        return response.status_code == 200 and response.headers.get(PID_HEADER) == str(pid)
    except httpx.TransportError:
        return False


def _wait_until_serving(process: subprocess.Popen[bytes], host: str, port: int) -> None:
    url: Final = metrics_url(host, port)
    deadline: Final = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
    with closing(HTTPHandler(timeout=_STARTUP_PROBE_TIMEOUT_SECONDS)) as http:
        while time.monotonic() < deadline:
            if (returncode := process.poll()) is not None:
                raise MetricsServerStartupError(
                    f"Prometheus metrics server exited with code {returncode} before serving {host}:{port}; "
                    "is the port already in use?"
                )
            if _answered_by(http, url, process.pid):
                return
            time.sleep(_STARTUP_POLL_INTERVAL_SECONDS)
    process.terminate()
    raise MetricsServerStartupError(
        f"Prometheus metrics server did not answer {url} within {_STARTUP_TIMEOUT_SECONDS:.0f}s"
    )


def start_metrics_server_process(host: str, port: int, multiproc_dir: str) -> subprocess.Popen[bytes]:
    """Spawn the metrics server next to the proxy and block until it answers on its port."""
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
    args: Final = _CliArgs.model_validate(vars(parser.parse_args(argv)))
    if not args.multiproc_dir:
        parser.error("--multiproc_dir or PROMETHEUS_MULTIPROC_DIR is required")
    run_metrics_server(host=args.host, port=args.port, multiproc_dir=args.multiproc_dir)


if __name__ == "__main__":
    main()
