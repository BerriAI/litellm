from __future__ import annotations

import json
import queue
import re
import socket
import subprocess
import sys
import threading
import time
from contextlib import closing
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Iterator, TextIO, cast

import httpx
import pytest
import yaml
from pydantic import BaseModel, ConfigDict

from litellm.rust_bridge import native_bridge_available

REPO_ROOT = Path(__file__).resolve().parents[3]

_SECRET_PATTERN = re.compile(r"(sk-[A-Za-z0-9_\-]+|Bearer\s+\S+)")

_CAPTURE_RESPONSE_BODY: bytes = json.dumps(
    {
        "pages": [{"index": 0, "markdown": "captured"}],
        "model": "mistral-ocr-latest",
        "usage_info": {"pages_processed": 1},
    }
).encode()

_LIVENESS_DEADLINE_SECONDS = 90.0
_LIVENESS_POLL_SECONDS = 0.5
_PROXY_TERMINATE_TIMEOUT_SECONDS = 15
_SERVER_JOIN_TIMEOUT_SECONDS = 5


class CaptureLitellmParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: str
    api_key: str
    api_base: str


class CaptureModelEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_name: str
    litellm_params: CaptureLitellmParams


class CaptureGeneralSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    master_key: str


class CaptureLitellmSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    drop_params: bool


class CaptureProxyConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_list: tuple[CaptureModelEntry, ...]
    general_settings: CaptureGeneralSettings
    litellm_settings: CaptureLitellmSettings


@dataclass(frozen=True)
class CaptureProxy:
    proxy_url: str
    master_key: str
    captures: queue.Queue[bytes]


def _sanitize(text: str) -> str:
    return _SECRET_PATTERN.sub("[redacted]", text)


def _make_capture_handler(
    captures: queue.Queue[bytes],
) -> type[BaseHTTPRequestHandler]:
    class _CaptureHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("content-length", "0"))
            captures.put(self.rfile.read(length))
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(_CAPTURE_RESPONSE_BODY)))
            self.end_headers()
            self.wfile.write(_CAPTURE_RESPONSE_BODY)

        def log_message(self, format: str, *args: object) -> None:
            return

    return _CaptureHandler


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        _, port = cast(tuple[str, int], sock.getsockname())
        return port


def _wait_for_liveness(base_url: str, deadline: float) -> bool:
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{base_url}/health/liveliness", timeout=2)
            if resp.status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(_LIVENESS_POLL_SECONDS)
    return False


def _capture_config(capture_port: int, master_key: str) -> CaptureProxyConfig:
    return CaptureProxyConfig(
        model_list=(
            CaptureModelEntry(
                model_name="rust-ocr-mistral-capture",
                litellm_params=CaptureLitellmParams(
                    model="mistral/mistral-ocr-latest",
                    api_key="sk-capture-test",
                    api_base=f"http://127.0.0.1:{capture_port}",
                ),
            ),
        ),
        general_settings=CaptureGeneralSettings(master_key=master_key),
        litellm_settings=CaptureLitellmSettings(drop_params=False),
    )


@pytest.fixture
def capture_proxy(tmp_path: Path) -> Iterator[CaptureProxy]:
    if not native_bridge_available():
        pytest.skip("compiled Rust OCR bridge is required for the capture E2E")

    master_key = "sk-1234"
    captures: queue.Queue[bytes] = queue.Queue()
    capture_server: HTTPServer | None = None
    server_thread: threading.Thread | None = None
    proxy: subprocess.Popen[bytes] | None = None
    proxy_log: TextIO | None = None
    proxy_log_path = tmp_path / "capture-proxy.log"
    try:
        capture_port = _free_port()
        capture_server = HTTPServer(
            ("127.0.0.1", capture_port), _make_capture_handler(captures)
        )
        server_thread = threading.Thread(
            target=capture_server.serve_forever, daemon=True
        )
        server_thread.start()

        proxy_port = _free_port()
        config_path = tmp_path / "capture-config.yml"
        config_path.write_text(
            yaml.safe_dump(_capture_config(capture_port, master_key).model_dump())
        )

        proxy_log = proxy_log_path.open("w")
        proxy = subprocess.Popen(
            [
                sys.executable,
                str(REPO_ROOT / "litellm" / "proxy" / "proxy_cli.py"),
                "--config",
                str(config_path),
                "--host",
                "127.0.0.1",
                "--port",
                str(proxy_port),
                "--num_workers",
                "1",
            ],
            cwd=str(REPO_ROOT),
            stdout=proxy_log,
            stderr=subprocess.STDOUT,
        )
        proxy_url = f"http://127.0.0.1:{proxy_port}"
        if not _wait_for_liveness(
            proxy_url, time.monotonic() + _LIVENESS_DEADLINE_SECONDS
        ):
            proxy_log.flush()
            tail = _sanitize(proxy_log_path.read_text()[-4000:])
            pytest.fail(
                f"capture proxy did not become live while the Rust bridge is available; "
                f"sanitized proxy log at {proxy_log_path}\n{tail}"
            )
        yield CaptureProxy(
            proxy_url=proxy_url, master_key=master_key, captures=captures
        )
    finally:
        if proxy is not None:
            proxy.terminate()
            try:
                proxy.wait(timeout=_PROXY_TERMINATE_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                proxy.kill()
                proxy.wait()
        if proxy_log is not None:
            proxy_log.close()
        if server_thread is not None and server_thread.ident is not None:
            if capture_server is not None:
                capture_server.shutdown()
            server_thread.join(timeout=_SERVER_JOIN_TIMEOUT_SECONDS)
        if capture_server is not None:
            capture_server.server_close()
