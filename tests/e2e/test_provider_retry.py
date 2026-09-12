import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Final


def test_retry_policy_preserves_failures_and_first_attempt_evidence(tmp_path: Path) -> None:
    suite: Final = Path(__file__).resolve().parent
    scenario: Final = tmp_path / "test_scenarios.py"
    scenario.write_text(
        """
from pathlib import Path
import json
from e2e_http import NetworkError, StreamingResponse, UnknownApiError, require_successful_call, unwrap

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
    attempt("stream_transport")
    require_successful_call(StreamingResponse(
        status_code=200, body="connection interrupted", network_error=NetworkError(message="connection interrupted"),
        headers={"x-amzn-requestid": "transport-request", "x-litellm-call-id": "transport-call"},
    ))
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
    assert "7 failed, 2 passed, 7 rerun" in result.stdout, result.stdout
    assert {path.stem: int(path.read_text()) for path in tmp_path.glob("*.attempts")} == {
        "recovers": 2,
        "persistent": 2,
        "proxy_503": 1,
        "auth": 1,
        "network": 2,
        "native": 2,
        "stream_recovers": 2,
        "stream_persistent": 2,
        "stream_transport": 2,
    }
    cases: Final = ET.parse(tmp_path / "results.xml").getroot()
    recovered: Final = cases.find(".//testcase[@name='test_recovers']")
    assert recovered is not None
    assert recovered.find("failure") is None
    assert recovered.find("properties/property[@name='upstream_request_id'][@value='aws-first-attempt']") is not None
    assert recovered.find("properties/property[@name='upstream_call_id'][@value='proxy-first-attempt']") is not None
    streamed: Final = cases.find(".//testcase[@name='test_stream_recovers']")
    assert streamed is not None and streamed.find("failure") is None
    assert streamed.find("properties/property[@name='upstream_status_code'][@value='200']") is not None
    assert streamed.find("properties/property[@name='upstream_evidence'][@value='stream_event']") is not None
    assert streamed.find("properties/property[@name='upstream_request_id'][@value='stream-request']") is not None
    assert streamed.find("properties/property[@name='upstream_call_id'][@value='stream-call']") is not None
    transport: Final = cases.find(".//testcase[@name='test_stream_transport']")
    assert transport is not None and transport.find("failure") is not None
    assert transport.find("properties/property[@name='upstream_request_id'][@value='transport-request']") is not None
    assert transport.find("properties/property[@name='upstream_call_id'][@value='transport-call']") is not None
