import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import litellm

pytestmark = pytest.mark.requires_rust_extension


@pytest.fixture
def ocr_server():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            response = json.dumps(
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

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield server, requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_ocr_with_rust_extension(ocr_server, monkeypatch):
    from litellm.ocr import main as ocr_main

    def reject_python_transport(**kwargs):
        pytest.fail("OCR used the Python transport instead of the Rust extension")

    monkeypatch.setattr(ocr_main.base_llm_http_handler, "ocr", reject_python_transport)
    server, requests = ocr_server
    host, port = server.server_address

    response = litellm.ocr(
        model="mistral/mistral-ocr-latest",
        document={"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
        api_key="test-key",
        api_base=f"http://{host}:{port}",
    )

    assert response.pages[0].markdown == "native OCR response"
    assert requests == [
        {
            "model": "mistral-ocr-latest",
            "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
        }
    ]
