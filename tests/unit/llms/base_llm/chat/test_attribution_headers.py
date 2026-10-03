"""
Provider attribution headers (`BaseConfig.get_attribution_headers`) must reach
the wire on every OpenAI-compatible chat path, and a caller header with the
same name must win.

These send real requests to a local server and assert on what it received,
because the default path (OpenAI SDK) never calls `validate_environment`.
"""

import json
import threading
from collections.abc import AsyncIterable, Iterable, Iterator, Mapping
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Final, cast

import pytest

import litellm
from litellm.llms.base_llm.chat.transformation import with_attribution_headers

_COMPLETION_BODY: Final = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "m",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
_STREAM_CHUNK: Final = {
    "id": "chatcmpl-1",
    "object": "chat.completion.chunk",
    "created": 1,
    "model": "m",
    "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
}


class _CaptureServer:
    def __init__(self) -> None:
        self.received: list[Mapping[str, list[str]]] = []
        captured: Final = self.received

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body: Final = json.loads(self.rfile.read(int(self.headers["content-length"])))
                lowered: dict[str, list[str]] = {}
                for name, value in self.headers.items():
                    lowered.setdefault(name.lower(), []).append(value)
                captured.append(lowered)
                if body.get("stream"):
                    payload = f"data: {json.dumps(_STREAM_CHUNK)}\n\ndata: [DONE]\n\n".encode()
                    content_type = "text/event-stream"
                else:
                    payload = json.dumps(_COMPLETION_BODY).encode()
                    content_type = "application/json"
                self.send_response(200)
                self.send_header("content-type", content_type)
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: object) -> None:
                pass

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.api_base = f"http://127.0.0.1:{self._server.server_port}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def last(self, header: str) -> list[str]:
        return self.received[-1].get(header.lower(), [])

    def close(self) -> None:
        self._server.shutdown()


@pytest.fixture
def server() -> Iterator[_CaptureServer]:
    capture: Final = _CaptureServer()
    yield capture
    capture.close()


@pytest.fixture(params=["sdk", "http_handler"])
def handler_path(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    if request.param == "http_handler":
        monkeypatch.setenv("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", "true")
    else:
        monkeypatch.delenv("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", raising=False)
    return request.param


_NOVITA_MODEL: Final = "novita/meta-llama/llama-3.3-70b-instruct"

# (model, attribution header name) for every provider that opts in
_ATTRIBUTED: Final = [
    pytest.param(_NOVITA_MODEL, "x-novita-source", id="novita"),
    pytest.param("perplexity/sonar", "x-pplx-integration", id="perplexity"),
]


def _drain(response: object) -> None:
    for _ in cast(Iterable[object], response):
        pass


async def _adrain(response: object) -> None:
    async for _ in cast(AsyncIterable[object], response):
        pass


@pytest.mark.parametrize(("model", "header"), _ATTRIBUTED)
@pytest.mark.parametrize("stream", [False, True])
def test_attribution_header_sent_sync(
    server: _CaptureServer, handler_path: str, model: str, header: str, stream: bool
) -> None:
    response: Final = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        api_base=server.api_base,
        api_key="k",
        stream=stream,
    )
    if stream:
        _drain(response)

    assert server.last(header) == ["litellm"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "header"), _ATTRIBUTED)
@pytest.mark.parametrize("stream", [False, True])
async def test_attribution_header_sent_async(
    server: _CaptureServer, handler_path: str, model: str, header: str, stream: bool
) -> None:
    response: Final = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        api_base=server.api_base,
        api_key="k",
        stream=stream,
    )
    if stream:
        await _adrain(response)

    assert server.last(header) == ["litellm"]


@pytest.mark.parametrize(("model", "header"), _ATTRIBUTED)
@pytest.mark.parametrize("header_kwarg", ["headers", "extra_headers"])
def test_caller_header_overrides_attribution_any_casing(
    server: _CaptureServer, handler_path: str, model: str, header: str, header_kwarg: str
) -> None:
    caller_headers: Final = {header.upper(): "my-app"}

    litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        api_base=server.api_base,
        api_key="k",
        **{header_kwarg: caller_headers},
    )

    assert server.last(header) == ["my-app"]
    assert caller_headers == {header.upper(): "my-app"}


def test_provider_without_attribution_sends_none(server: _CaptureServer, handler_path: str) -> None:
    litellm.completion(
        model="together_ai/meta-llama/Llama-3-8b-chat-hf",
        messages=[{"role": "user", "content": "hi"}],
        api_base=server.api_base,
        api_key="k",
    )

    assert server.last("x-novita-source") == []
    assert server.last("x-pplx-integration") == []


def test_global_litellm_headers_still_apply_and_are_not_mutated(
    server: _CaptureServer, handler_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    global_headers: Final = {"X-Global": "1"}
    monkeypatch.setattr(litellm, "headers", global_headers)

    litellm.completion(
        model=_NOVITA_MODEL,
        messages=[{"role": "user", "content": "hi"}],
        api_base=server.api_base,
        api_key="k",
    )

    assert server.last("x-global") == ["1"]
    assert server.last("x-novita-source") == ["litellm"]
    assert global_headers == {"X-Global": "1"}


def test_with_attribution_headers_returns_headers_unchanged_when_nothing_to_add() -> None:
    headers: Final = {"A": "1"}

    assert with_attribution_headers({}, headers) is headers
    assert with_attribution_headers({}, None) is None
