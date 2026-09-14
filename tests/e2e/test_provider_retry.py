import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Final


def test_retry_policy_does_not_turn_new_diagnostics_into_recovered_passes(tmp_path: Path) -> None:
    suite: Final = Path(__file__).resolve().parent
    scenario: Final = tmp_path / "test_scenarios.py"
    scenario.write_text(
        """
from pathlib import Path
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import pytest
from e2e_http import URL, NoBody, NetworkError, StreamingResponse, UnknownApiError, require_successful_call, send, unwrap
from test_bedrock_eventstream import _CONVERSE, _frame
from llm_translation.bedrock_stream import assert_converse_stream

def attempt(name):
    path = Path(__file__).with_name(name + ".attempts")
    count = int(path.read_text()) + 1 if path.exists() else 1
    path.write_text(str(count))
    return count

def unavailable():
    body = json.dumps({"error":{"message": "litellm.ServiceUnavailableError: BedrockException - Bedrock is unable to process your request."}})
    unwrap(UnknownApiError(status_code=503, body=body, headers={
        "x-amzn-requestid": "aws-first-attempt", "x-amzn-errortype": "ServiceUnavailableException",
        "x-litellm-call-id": "proxy-first-attempt",
    }))

def test_recovers():
    if attempt("recovers") == 1:
        unavailable()

def test_persistent():
    attempt("persistent")
    unavailable()

def test_proxy_503():
    attempt("proxy_503")
    unwrap(UnknownApiError(status_code=503, body="proxy overloaded"))

def test_auth():
    attempt("auth")
    unwrap(UnknownApiError(status_code=403, body="invalid credentials"))

def test_network():
    attempt("network")
    unwrap(NetworkError(message="connection reset"))

def test_native():
    attempt("native")
    require_successful_call(StreamingResponse(status_code=503, body="unavailable", headers={
        "x-amzn-requestid": "native-request", "x-amzn-errortype": "ServiceUnavailableException",
    }), expected_provider="bedrock")

def streamed_unavailable():
    require_successful_call(StreamingResponse(
        status_code=200, body="<streamed>", stream_error="serviceUnavailableException: unavailable",
        stream_error_code="serviceUnavailableException", headers={
            "x-amzn-requestid": "stream-request", "x-litellm-call-id": "stream-call",
        },
    ), expected_provider="bedrock")

def test_stream_recovers():
    if attempt("stream_recovers") == 1:
        streamed_unavailable()

def test_stream_persistent():
    attempt("stream_persistent")
    streamed_unavailable()

def test_stream_transport():
    if attempt("stream_transport") == 1:
        require_successful_call(StreamingResponse(
            status_code=200, body="connection interrupted", network_error=NetworkError(message="kind='network'"),
            headers={"x-amzn-requestid": "transport-request", "x-litellm-call-id": "transport-call"},
        ))

def test_connection_recovers():
    if attempt("connection_recovers") == 1:
        require_successful_call(StreamingResponse(
            status_code=-1, body="connection refused", network_error=NetworkError(message="kind='network'"),
        ))

def test_connection_direct_recovers():
    if attempt("connection_direct_recovers") == 1:
        result = StreamingResponse(status_code=-1, body="connection refused", network_error=NetworkError(message="connection refused"))
        assert result.ok, f"call failed: {result}"

def test_header_marker_recovers():
    if attempt("header_marker_recovers") == 1:
        require_successful_call(StreamingResponse(status_code=403, body="invalid credentials", headers={"x-request-id": "kind='network'"}))

def test_behavior_recovers():
    assert attempt("behavior_recovers") > 1, "completion returned no content"

def test_healthy_control():
    require_successful_call(StreamingResponse(status_code=200, body="{}"))

class InterruptedOnce(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", "0")))
        binary = self.path.endswith("-binary")
        wire = (
            b"".join(_frame(json.dumps(value), key) for payload in _CONVERSE for key, value in json.loads(payload).items())
            if binary else b'data: {"text":"Hello"}\\n\\ndata: [DONE]\\n\\n'
        )
        interrupted = attempt("http_" + self.path.lstrip("/")) == 1 and not self.path.startswith("/healthy-")
        data = wire[:30] if interrupted else wire
        self.send_response(200)
        self.send_header("content-type", "application/vnd.amazon.eventstream" if binary else "text/event-stream")
        self.send_header("transfer-encoding", "chunked")
        self.send_header("x-litellm-call-id", "controlled-disconnect")
        self.end_headers()
        self.wfile.write(f"{len(data):x}\\r\\n".encode() + data + b"\\r\\n")
        if not interrupted:
            self.wfile.write(b"0\\r\\n\\r\\n")
        self.wfile.flush()
        self.close_connection = True

    def log_message(self, *args):
        pass

@pytest.fixture(scope="session")
def endpoint():
    server = ThreadingHTTPServer(("127.0.0.1", 0), InterruptedOnce)
    worker = Thread(target=server.serve_forever, daemon=True)
    worker.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    worker.join(timeout=5)

@pytest.mark.parametrize("path", ["binary", "sse"])
@pytest.mark.parametrize("mode", ["helper", "direct", "healthy"])
def test_http_stream_would_recover_on_a_second_request(endpoint, path, mode):
    result = send(URL(f"{endpoint}/{mode}-{path}"), headers=NoBody(), json=NoBody(), stream=True, timeout=5)
    if mode == "direct":
        assert result.ok, f"stream was not established: {result}"
    else:
        require_successful_call(result)
    if path == "binary":
        assert_converse_stream(result)
    else:
        assert result.stream_done
""",
        encoding="utf-8",
    )
    result: Final = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(scenario),
            "-c",
            str(suite / "pytest.ini"),
            "--rootdir",
            str(tmp_path),
            "--confcutdir",
            str(tmp_path),
            "-p",
            "conftest",
            "--reruns-delay",
            "0",
            "--junitxml",
            str(tmp_path / "results.xml"),
            "-q",
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(suite), "PYTEST_ADDOPTS": "", "E2E_FIXTURE_MODE": "live"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "17 failed, 3 passed, 1 rerun" in result.stdout, result.stdout
    assert {path.stem: int(path.read_text()) for path in tmp_path.glob("*.attempts")} == {
        "recovers": 1,
        "persistent": 1,
        "proxy_503": 1,
        "auth": 1,
        "network": 2,
        "native": 1,
        "stream_recovers": 1,
        "stream_persistent": 1,
        "stream_transport": 1,
        "connection_recovers": 1,
        "behavior_recovers": 1,
        "connection_direct_recovers": 1,
        "header_marker_recovers": 1,
        "http_helper-binary": 1,
        "http_helper-sse": 1,
        "http_direct-binary": 1,
        "http_direct-sse": 1,
        "http_healthy-binary": 1,
        "http_healthy-sse": 1,
    }
    cases: Final = ET.parse(tmp_path / "results.xml").getroot()
    recovered: Final = cases.find(".//testcase[@name='test_recovers']")
    assert recovered is not None
    assert recovered.find("failure") is not None
    assert recovered.find("properties/property[@name='upstream_request_id'][@value='aws-first-attempt']") is not None
    assert recovered.find("properties/property[@name='upstream_call_id'][@value='proxy-first-attempt']") is not None
    streamed: Final = cases.find(".//testcase[@name='test_stream_recovers']")
    assert streamed is not None and streamed.find("failure") is not None
    assert streamed.find("properties/property[@name='upstream_status_code'][@value='200']") is not None
    assert streamed.find("properties/property[@name='upstream_evidence'][@value='stream_event']") is not None
    assert streamed.find("properties/property[@name='upstream_request_id'][@value='stream-request']") is not None
    assert streamed.find("properties/property[@name='upstream_call_id'][@value='stream-call']") is not None
    transport: Final = cases.find(".//testcase[@name='test_stream_transport']")
    assert transport is not None and transport.find("failure") is not None
    assert transport.find("properties/property[@name='upstream_request_id'][@value='transport-request']") is not None
    assert transport.find("properties/property[@name='upstream_call_id'][@value='transport-call']") is not None
