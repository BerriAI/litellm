"""`max_retries` passed to image_generation() reaches the OpenAI image handler.

It used to be filtered out with the other OpenAI client params and never
forwarded, so the handler always fell back to its default of 2 retries.
"""

from unittest.mock import patch

import pytest

import litellm
from litellm.images.main import image_generation


def _optional_params(mock_openai_chat_completions) -> dict:
    call_kwargs = mock_openai_chat_completions.image_generation.call_args
    return call_kwargs.kwargs["optional_params"]


@pytest.mark.parametrize("max_retries", [0, 1, 5])
@patch("litellm.images.main.openai_chat_completions")
def test_max_retries_is_forwarded_to_the_openai_image_handler(mock_openai_chat_completions, max_retries):
    mock_openai_chat_completions.image_generation.return_value = litellm.utils.ImageResponse(
        created=1234567890, data=[{"url": "https://example.com/image.png"}]
    )

    image_generation(model="openai/dall-e-3", prompt="A red circle", max_retries=max_retries)

    assert _optional_params(mock_openai_chat_completions)["max_retries"] == max_retries


@patch("litellm.images.main.openai_chat_completions")
def test_max_retries_is_absent_when_not_provided(mock_openai_chat_completions):
    """The handler keeps its own default when the caller sets nothing."""
    mock_openai_chat_completions.image_generation.return_value = litellm.utils.ImageResponse(
        created=1234567890, data=[{"url": "https://example.com/image.png"}]
    )

    image_generation(model="openai/dall-e-3", prompt="A red circle")

    assert "max_retries" not in _optional_params(mock_openai_chat_completions)


def test_max_retries_zero_makes_one_upstream_request():
    """End to end through the real OpenAI handler, against a local stub that 503s."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    hits = []

    class Stub(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length", 0) or 0))
            hits.append(self.path)
            self.send_response(503)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": {"message": "unavailable", "type": "server_error"}}')

    server = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with pytest.raises(Exception):
            image_generation(
                model="openai/dall-e-3",
                prompt="A red circle",
                api_base=f"http://127.0.0.1:{server.server_address[1]}/v1",
                api_key="sk-test",
                max_retries=0,
            )
    finally:
        server.shutdown()

    assert len(hits) == 1
