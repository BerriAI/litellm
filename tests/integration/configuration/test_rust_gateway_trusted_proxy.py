import http.client
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest
from pydantic import TypeAdapter

from tests.integration._support.ports import allocate_port

ROOT: Final = Path(__file__).resolve().parents[3]
DEFAULT_BINARY: Final = ROOT / "litellm-rust" / "target" / "debug" / "litellm-gateway"
HEADER_OBSERVATION: Final = TypeAdapter(dict[str, str | None])
APP_SOURCE: Final = """\
from fastapi import FastAPI, Request

app = FastAPI()

@app.get("/health/readiness")
async def readiness():
    return {"status": "healthy"}

@app.get("/headers")
async def headers(request: Request):
    return {
        "client": request.client.host if request.client is not None else None,
        "host": request.headers.get("host"),
        "origin": request.headers.get("origin"),
        "xff": request.headers.get("x-forwarded-for"),
        "proto": request.headers.get("x-forwarded-proto"),
        "scheme": request.url.scheme,
    }
"""


def _request(address: str, port: int, source: str | None = None) -> dict[str, str | None]:
    client: Final = http.client.HTTPConnection(
        address,
        port,
        timeout=1,
        source_address=None if source is None else (source, 0),
    )
    try:
        client.request(
            "GET",
            "/headers",
            headers={
                "Host": "gateway.example",
                "Origin": "https://ui.example",
                "X-Forwarded-For": "198.51.100.7",
                "X-Forwarded-Proto": "https",
                "Connection": "close",
            },
        )
        response: Final = client.getresponse()
        body: Final = response.read()
        assert response.status == 200, body
        return HEADER_OBSERVATION.validate_json(body)
    finally:
        client.close()


def _wait_until_ready(address: str, port: int, process: subprocess.Popen[bytes]) -> None:
    deadline: Final = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            _stdout, stderr = process.communicate()
            pytest.fail(
                f"Rust gateway exited during startup with status {process.returncode}: "
                f"{(stderr or b'').decode(errors='replace')}"
            )
        connection: Final = http.client.HTTPConnection(address, port, timeout=0.2)
        try:
            connection.request("GET", "/health/readiness")
            response: Final = connection.getresponse()
            response.read()
            if response.status == 200:
                return
        except OSError:
            pass
        finally:
            connection.close()
        time.sleep(0.05)
    pytest.fail("Rust gateway did not forward readiness to the Uvicorn fixture")


@pytest.mark.parametrize(
    ("trusted_proxies", "expected_client", "expected_proto"),
    (
        ("", "127.0.0.2", "http"),
        ("127.0.0.2", "198.51.100.7", "https"),
    ),
)
def test_rust_gateway_canonicalizes_headers_before_real_uvicorn(
    tmp_path: Path,
    trusted_proxies: str,
    expected_client: str,
    expected_proto: str,
) -> None:
    binary: Final = Path(os.environ.get("LITELLM_RUST_GATEWAY_BINARY", str(DEFAULT_BINARY)))
    if not binary.is_file():
        pytest.skip("build litellm-rust/crates/gateway before running the real Uvicorn trust fixture")

    python_port: Final = allocate_port()
    rust_port: Final = allocate_port()
    app_path: Final = tmp_path / "fixture.py"
    app_path.write_text(APP_SOURCE)
    python_environment: Final = {
        **os.environ,
        "PYTHONPATH": str(tmp_path),
        "FORWARDED_ALLOW_IPS": "127.0.0.1",
    }
    python: Final = subprocess.Popen(
        (
            sys.executable,
            "-P",
            "-m",
            "uvicorn",
            "fixture:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(python_port),
            "--proxy-headers",
            "--forwarded-allow-ips",
            "127.0.0.1",
        ),
        env=python_environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    _wait_until_ready("127.0.0.1", python_port, python)
    direct_python: Final = _request("127.0.0.1", python_port, "127.0.0.2")
    assert direct_python == {
        "client": "127.0.0.2",
        "host": "gateway.example",
        "origin": "https://ui.example",
        "xff": "198.51.100.7",
        "proto": "https",
        "scheme": "http",
    }
    rust_environment: Final[Mapping[str, str]] = {
        **os.environ,
        "HOST": "127.0.0.1",
        "PORT": str(rust_port),
        "LITELLM_RUST_GATEWAY": "true",
        "LITELLM_PYTHON_UPSTREAM": f"http://127.0.0.1:{python_port}",
        "LITELLM_RUST_GATEWAY_TRUSTED_PROXIES": trusted_proxies,
    }
    rust: Final = subprocess.Popen(
        (str(binary),),
        env=rust_environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        _wait_until_ready("127.0.0.1", rust_port, rust)
        response: Final = _request("127.0.0.1", rust_port, "127.0.0.2")

        assert response == {
            "client": expected_client,
            "host": "gateway.example",
            "origin": "https://ui.example",
            "xff": expected_client,
            "proto": expected_proto,
            "scheme": expected_proto,
        }
    finally:
        for process in (rust, python):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)  # test-quality-ok: Signal only this fixture's process group
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)  # test-quality-ok: Kill only this fixture's process group
                    process.wait(timeout=5)
