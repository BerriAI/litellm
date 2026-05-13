import asyncio
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Tuple

from scripts.cloud_agent_perf import (
    EndpointConfig,
    build_payload,
    compare_overhead,
    main,
    run_load_test,
)


def _start_mock_chat_server() -> Tuple[ThreadingHTTPServer, str, List[dict]]:
    requests: List[dict] = []

    class MockChatHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length)
            requests.append(json.loads(body))
            response_body = json.dumps(
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "hello"},
                            "finish_reason": "stop",
                        }
                    ],
                }
            ).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), MockChatHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    server.thread = thread  # type: ignore[attr-defined]
    url = f"http://127.0.0.1:{server.server_address[1]}/chat/completions"
    return server, url, requests


def test_should_measure_rps_against_mock_chat_endpoint() -> None:
    server, url, requests = _start_mock_chat_server()
    try:
        payload = build_payload(model="fake-openai-endpoint")
        result = asyncio.run(
            run_load_test(
                endpoint=EndpointConfig(
                    name="mock proxy",
                    url=url,
                    model="fake-openai-endpoint",
                    api_key="sk-test",
                ),
                payload=payload,
                requests=4,
                concurrency=2,
                warmup_requests=1,
                timeout_s=5,
            )
        )
    finally:
        server.shutdown()
        server.server_close()
        server.thread.join(timeout=5)  # type: ignore[attr-defined]

    summary = result.to_summary()
    assert summary["requests"] == 4
    assert summary["successful_requests"] == 4
    assert summary["failed_requests"] == 0
    assert summary["rps"] > 0
    assert summary["successful_rps"] > 0
    assert summary["latency_ms"]["mean"] > 0
    assert len(requests) == 5
    assert all(request["model"] == "fake-openai-endpoint" for request in requests)


def test_should_calculate_proxy_overhead_from_latency_summaries() -> None:
    overhead = compare_overhead(
        proxy_summary={
            "latency_ms": {"mean": 15.0, "p50": 12.0, "p95": 25.0, "p99": 40.0}
        },
        direct_summary={
            "latency_ms": {"mean": 10.0, "p50": 8.0, "p95": 20.0, "p99": 32.0}
        },
    )

    assert overhead["mean_overhead_ms"] == 5.0
    assert overhead["mean_overhead_pct"] == 50.0
    assert overhead["p95_overhead_ms"] == 5.0
    assert overhead["p99_overhead_pct"] == 25.0


def test_should_run_cli_against_mock_chat_endpoint(
    tmp_path, monkeypatch, capsys
) -> None:
    server, url, requests = _start_mock_chat_server()
    output_path = tmp_path / "perf.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "cloud_agent_perf.py",
            "--proxy-url",
            url,
            "--proxy-api-key",
            "sk-test",
            "--model",
            "fake-openai-endpoint",
            "--requests",
            "2",
            "--concurrency",
            "1",
            "--warmup-requests",
            "0",
            "--output-json",
            str(output_path),
        ],
    )
    try:
        exit_code = main()
    finally:
        server.shutdown()
        server.server_close()
        server.thread.join(timeout=5)  # type: ignore[attr-defined]

    output = capsys.readouterr().out
    data = json.loads(output_path.read_text())
    assert exit_code == 0
    assert "rps:" in output
    assert "Wrote JSON results" in output
    assert data["proxy"]["successful_requests"] == 2
    assert len(requests) == 2
