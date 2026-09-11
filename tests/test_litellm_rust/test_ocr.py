import json
import threading
from collections.abc import Generator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final

import pytest

from litellm.rust_bridge import ocr as rust_ocr_bridge

pytestmark = pytest.mark.requires_rust_extension


@dataclass(frozen=True, slots=True)
class RecordedOCRRequest:
    body: object


@pytest.fixture
def ocr_server() -> Generator[tuple[ThreadingHTTPServer, list[RecordedOCRRequest]]]:
    requests: Final[list[RecordedOCRRequest]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            requests.append(
                RecordedOCRRequest(
                    body=json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                )
            )
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
    ocr_server: tuple[ThreadingHTTPServer, list[RecordedOCRRequest]],
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
    assert requests[0].body == {
        "model": "mistral-ocr-latest",
        "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
    }
