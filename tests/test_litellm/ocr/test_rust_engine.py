"""
Integration tests for the native Rust OCR engine (`litellm_rust`).

Mistral OCR is served by the `litellm_rust` extension: provider routing,
request/response translation and the upstream HTTP call all happen in Rust,
while auth / logging / spend stay in Python via the `@client` decorator.

These tests stand up a fake Mistral OCR server and drive `litellm.ocr` /
`litellm.aocr` through the real Rust path (no network egress, no API key). They
self-skip when the `litellm_rust` wheel is not built (e.g. in CI jobs that don't
compile the crate); build it with `maturin develop --features python` from
`litellm-rust/`.
"""

import asyncio
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import litellm

# Skip the whole module if the native engine isn't installed.
pytest.importorskip("litellm_rust")

# Canonical Mistral OCR response shape (the standard LiteLLM OCR format).
GOLDEN_RESPONSE = {
    "pages": [
        {
            "index": 0,
            "markdown": "# Title\nSome extracted text.",
            "images": [],
            "dimensions": {"dpi": 200, "height": 2200, "width": 1700},
        }
    ],
    "model": "mistral-ocr-2505-completion",
    "document_annotation": None,
    "usage_info": {"pages_processed": 1, "doc_size_bytes": 12345},
}

DOCUMENT = {"type": "document_url", "document_url": "https://example.com/doc.pdf"}


class _FakeMistral:
    """A throwaway HTTP server that mimics Mistral's POST /v1/ocr endpoint."""

    def __init__(self, status: int = 200):
        self.status = status
        self.captured: dict = {}
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):  # silence
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                server.captured = {
                    "path": self.path,
                    "authorization": self.headers.get("Authorization"),
                    "body": body,
                }
                if server.status != 200:
                    payload = json.dumps({"error": "boom"}).encode()
                else:
                    payload = json.dumps(GOLDEN_RESPONSE).encode()
                self.send_response(server.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"

    def __enter__(self):
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._httpd.shutdown()


def test_ocr_sync_routes_through_rust_engine():
    """litellm.ocr() for a mistral model goes through Rust and returns OCRResponse."""
    with _FakeMistral() as server:
        response = litellm.ocr(
            model="mistral/mistral-ocr-latest",
            document=DOCUMENT,
            api_base=server.base,
            api_key="fake-key-123",
            include_image_base64=True,
            pages=[0],
        )

    # Response parsed into the standard OCRResponse.
    assert type(response).__name__ == "OCRResponse"
    assert response.model == GOLDEN_RESPONSE["model"]
    assert response.pages[0].index == 0
    assert response.pages[0].markdown == GOLDEN_RESPONSE["pages"][0]["markdown"]
    assert response.usage_info.pages_processed == 1

    # The Rust engine built the request faithfully.
    sent = server.captured
    assert sent["path"] == "/v1/ocr"
    assert sent["authorization"] == "Bearer fake-key-123"
    assert sent["body"]["model"] == "mistral-ocr-latest"  # provider stripped
    assert sent["body"]["document"] == DOCUMENT
    assert sent["body"]["include_image_base64"] is True
    assert sent["body"]["pages"] == [0]


def test_aocr_async_routes_through_rust_engine():
    """The async path (run in executor) also returns a valid OCRResponse."""
    with _FakeMistral() as server:
        response = asyncio.run(
            litellm.aocr(
                model="mistral/mistral-ocr-latest",
                document=DOCUMENT,
                api_base=server.base,
                api_key="fake-key-123",
            )
        )
    assert type(response).__name__ == "OCRResponse"
    assert response.model == GOLDEN_RESPONSE["model"]
    assert len(response.pages) == 1


def test_unsupported_params_are_filtered_by_rust():
    """Params outside the supported OCR set are dropped before the upstream call."""
    with _FakeMistral() as server:
        litellm.ocr(
            model="mistral/mistral-ocr-latest",
            document=DOCUMENT,
            api_base=server.base,
            api_key="k",
            pages=[1, 2],
            not_a_real_param="should_be_dropped",
        )
    assert "not_a_real_param" not in server.captured["body"]
    assert server.captured["body"]["pages"] == [1, 2]


def test_upstream_error_propagates():
    """A non-2xx upstream response surfaces as an exception, not a silent None."""
    with _FakeMistral(status=500) as server:
        with pytest.raises(Exception):
            litellm.ocr(
                model="mistral/mistral-ocr-latest",
                document=DOCUMENT,
                api_base=server.base,
                api_key="k",
            )


@pytest.mark.skipif(
    not os.environ.get("RUN_LIVE_MISTRAL_OCR"),
    reason="Set RUN_LIVE_MISTRAL_OCR=1 (with a valid MISTRAL_API_KEY) to run the live call",
)
def test_ocr_live_mistral():
    """Live end-to-end call against the real Mistral OCR API (opt-in)."""
    response = litellm.ocr(
        model="mistral/mistral-ocr-latest",
        document={
            "type": "document_url",
            "document_url": "https://arxiv.org/pdf/2201.04234",
        },
    )
    assert type(response).__name__ == "OCRResponse"
    assert len(response.pages) > 0
    assert response.pages[0].markdown
