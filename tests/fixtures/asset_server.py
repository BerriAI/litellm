"""Serves tests/fixtures over loopback HTTP.

Some code paths under test (``convert_url_to_base64``, Bedrock's image
embedding, Gemini's tool-result media handling) only run when the image arrives
as a URL, so a ``data:`` URL would skip the very branch being tested. Those
tests point at this server instead of a third-party image host.

litellm's SSRF guard rejects loopback by default, so the server also adds its
own host to ``litellm.user_url_allowed_hosts`` for as long as it is up: the
same setting an operator uses to reach an internal image host.
"""

from __future__ import annotations

import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

import litellm

ASSETS_DIR = Path(__file__).parent
TEST_IMAGE_PNG = ASSETS_DIR / "test_image.png"
TEST_IMAGE_JPG = ASSETS_DIR / "test_image.jpg"
TEST_SPEECH_WAV = ASSETS_DIR / "test_speech.wav"
TEST_DOCUMENT_PDF = ASSETS_DIR / "test_document.pdf"
TEST_DOCUMENT_MD = ASSETS_DIR / "test_document.md"
TEST_TABLE_CSV = ASSETS_DIR / "test_table.csv"


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args: object) -> None:
        pass


def serve_assets() -> Iterator[str]:
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(_QuietHandler, directory=str(ASSETS_DIR))
    )
    host = f"127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    previously_allowed = litellm.user_url_allowed_hosts
    litellm.user_url_allowed_hosts = [*previously_allowed, host]
    try:
        yield f"http://{host}"
    finally:
        litellm.user_url_allowed_hosts = previously_allowed
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
