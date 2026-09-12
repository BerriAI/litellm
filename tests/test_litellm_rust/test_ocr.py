import json
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final

import pytest

import litellm
from litellm.rust_bridge import ocr as rust_ocr_bridge

pytestmark = pytest.mark.requires_rust_extension


@pytest.fixture
def ocr_server() -> Generator[tuple[ThreadingHTTPServer, list[dict[str, object]]]]:
    requests: Final[list[dict[str, object]]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            requests.append(
                {
                    "headers": {name.lower(): value for name, value in self.headers.items()},
                    "body": json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                }
            )
            if self.headers.get("x-test-stall") == "true":
                self.connection.settimeout(2)
                try:
                    self.rfile.read(1)
                except TimeoutError:
                    pass
                return
            if self.headers.get("User-Agent", "").startswith("python-httpx"):
                self.send_response(418)
                self.end_headers()
                return
            status = int(self.headers.get("x-test-status", "200"))
            if status != 200:
                body = b'{"error":"provider unavailable"}'
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            response: Final = json.dumps(
                {
                    "pages": [{"index": 0, "markdown": "native OCR response", "images": [], "dimensions": None}],
                    "model": "mistral-ocr-latest",
                    "usage_info": {"pages_processed": 1, "doc_size_bytes": 3},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread: Final = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield server, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_native_ocr_with_compiled_rust_extension(
    ocr_server: tuple[ThreadingHTTPServer, list[dict[str, object]]],
) -> None:
    server, requests = ocr_server
    address: Final = server.server_address
    host: Final = str(address[0])
    port: Final = int(address[1])

    response: Final = rust_ocr_bridge.ocr(
        model="mistral-ocr-latest",
        document={"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
        api_key="test-key",
        api_base=f"http://{host}:{port}",
        custom_llm_provider="mistral",
        extra_headers=None,
        optional_params={},
        timeout=None,
    )

    assert response is not None
    assert response["pages"][0]["markdown"] == "native OCR response"
    assert len(requests) == 1
    assert not requests[0]["headers"].get("user-agent", "").startswith("python-httpx")
    assert requests[0]["body"] == {
        "model": "mistral-ocr-latest",
        "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
    }


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("model", ["mistral/mistral-ocr-latest", "azure_ai/doc-intelligence/prebuilt-read"])
@pytest.mark.asyncio
async def test_native_public_ocr_matches_python(model, asynchronous):
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    from typing import Final
    from urllib.parse import parse_qsl, urlsplit

    from litellm.rust_bridge import _native

    assert callable(_native.ocr)
    calls: Final = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body: Final = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            target: Final = urlsplit(self.path)
            calls.append(
                (
                    target.path,
                    parse_qsl(target.query),
                    self.headers.get("Authorization"),
                    self.headers.get("Ocp-Apim-Subscription-Key"),
                    body,
                )
            )
            payload: Final = (
                {"status": "succeeded", "analyzeResult": {"pages": []}}
                if "doc-intelligence" in model
                else {"pages": [{"index": 0, "markdown": "hello"}]}
            )
            encoded: Final = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_args):
            pass

    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread: Final = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    responses: Final = []
    try:
        for enabled in (False, True):
            litellm.rust(enabled)
            arguments: Final = {
                "model": model,
                "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
                "api_key": "test-key",
                "api_base": f"http://127.0.0.1:{server.server_port}",
                "pages": [0, 2],
                "timeout": 3.0,
            }
            response: Final = await litellm.aocr(**arguments) if asynchronous else litellm.ocr(**arguments)
            responses.append(response.model_dump())
        assert len(calls) == 2
        assert calls[0] == calls[1]
        for key in ("model", "pages", "object"):
            assert responses[0][key] == responses[1][key]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_native_ocr_failures_do_not_retry_on_python(ocr_server, asynchronous):
    server, requests = ocr_server
    arguments = {
        "model": "mistral-ocr-latest",
        "custom_llm_provider": "mistral",
        "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
        "api_key": "test-key",
        "api_base": f"http://127.0.0.1:{server.server_port}",
        "extra_headers": {"x-test-status": "503"},
        "num_retries": 0,
    }
    litellm.rust(True)
    with pytest.raises(litellm.ServiceUnavailableError) as caught:
        await litellm.aocr(**arguments) if asynchronous else litellm.ocr(**arguments)
    assert caught.value.status_code == 503
    assert len(requests) == 1
    assert not requests[0]["headers"].get("user-agent", "").startswith("python-httpx")


@pytest.mark.parametrize("custom_provider", ["mistral", "not-a-provider"])
def test_native_ocr_rejects_invalid_input_before_network(ocr_server, custom_provider):
    from litellm.rust_bridge import _native

    server, requests = ocr_server
    with pytest.raises(ValueError, match=r"invalid (OCR request field|provider)|invalid request"):
        _native.ocr(
            model="mistral-ocr-latest",
            custom_llm_provider=custom_provider,
            document={"type": "document_url"},
            api_key="test-key",
            api_base=f"http://127.0.0.1:{server.server_port}",
        )
    assert requests == []


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_native_ocr_enforces_request_deadline_without_fallback(ocr_server, asynchronous):
    import asyncio
    import time

    server, requests = ocr_server
    litellm.rust(True)
    arguments = {
        "model": "mistral/mistral-ocr-latest",
        "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
        "api_key": "test-key",
        "api_base": f"http://127.0.0.1:{server.server_port}",
        "extra_headers": {"x-test-stall": "true"},
        "timeout": 0.1,
        "num_retries": 0,
    }
    started = time.monotonic()
    with pytest.raises(litellm.APIConnectionError):
        await asyncio.wait_for(
            litellm.aocr(**arguments) if asynchronous else asyncio.to_thread(litellm.ocr, **arguments),
            timeout=3,
        )
    assert 0.09 <= time.monotonic() - started < 3
    assert len(requests) == 1
    assert not requests[0]["headers"].get("user-agent", "").startswith("python-httpx")
