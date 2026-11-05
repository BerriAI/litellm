import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import litellm


class _Gemini400Server(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        body = b'{"error": {"code": 400, "message": "bad request", "status": "INVALID_ARGUMENT"}}'
        self.send_response(400)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def gemini_400_url():
    server = HTTPServer(("127.0.0.1", 0), _Gemini400Server)
    port = server.server_port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def test_router_sync_streaming_gemini_error_mapped(gemini_400_url):
    """
    Issue #49933: Router.completion(stream=True) with a Gemini deployment raising
    HTTP 400 should map to litellm.BadRequestError instead of TypeError (can't concat str to bytes)
    or raw MaskedHTTPStatusError.
    """
    params = {
        "model": "gemini/gemini-2.5-flash",
        "api_base": gemini_400_url,
        "api_key": "fake-key",
    }
    router = litellm.Router(
        model_list=[{"model_name": "test-gemini", "litellm_params": params}],
        num_retries=0,
    )

    with pytest.raises(litellm.BadRequestError) as exc_info:
        for _ in router.completion(
            model="test-gemini",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        ):
            pass

    assert isinstance(exc_info.value, litellm.BadRequestError)


@pytest.mark.asyncio
async def test_router_async_streaming_gemini_error_mapped(gemini_400_url):
    """
    Issue #49933: Router.acompletion(stream=True) with a Gemini deployment raising
    HTTP 400 should map to litellm.BadRequestError instead of raw VertexAIError.
    """
    params = {
        "model": "gemini/gemini-2.5-flash",
        "api_base": gemini_400_url,
        "api_key": "fake-key",
    }
    router = litellm.Router(
        model_list=[{"model_name": "test-gemini", "litellm_params": params}],
        num_retries=0,
    )

    with pytest.raises(litellm.BadRequestError) as exc_info:
        stream = await router.acompletion(
            model="test-gemini",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        async for _ in stream:
            pass

    assert isinstance(exc_info.value, litellm.BadRequestError)
