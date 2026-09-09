import json
import threading
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final

import pytest

import litellm

pytestmark = pytest.mark.requires_rust_extension


@pytest.fixture
def native_only_ocr_server() -> Generator[tuple[ThreadingHTTPServer, list[dict[str, object]]]]:
    requests: Final[list[dict[str, object]]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            requests.append(
                {
                    "headers": {name.lower(): value for name, value in self.headers.items()},
                    "body": json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                }
            )
            if self.headers.get("User-Agent", "").startswith("python-httpx"):
                self.send_response(418)
                self.end_headers()
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


def test_public_ocr_executes_the_compiled_extension_without_python_fallback(
    native_only_ocr_server: tuple[ThreadingHTTPServer, list[dict[str, object]]],
) -> None:
    server, requests = native_only_ocr_server
    address: Final = server.server_address
    host: Final = str(address[0])
    port: Final = int(address[1])

    response: Final = litellm.ocr(
        model="mistral/mistral-ocr-latest",
        document={"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
        api_key="test-key",
        api_base=f"http://{host}:{port}",
    )

    assert response.pages[0].markdown == "native OCR response"
    assert len(requests) == 1
    headers: Final = requests[0]["headers"]
    assert isinstance(headers, dict)
    assert not str(headers.get("user-agent", "")).startswith("python-httpx")
    assert requests[0]["body"] == {
        "model": "mistral-ocr-latest",
        "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
    }
