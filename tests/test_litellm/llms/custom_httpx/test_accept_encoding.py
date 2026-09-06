import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import httpx._client
import pytest

import litellm
from litellm.experimental_mcp_client.client import MCPClient
from litellm.llms.base import BaseLLM
from litellm.llms.custom_httpx.accept_encoding import (
    ACCEPT_ENCODING_OVERRIDE_ENV_VAR,
    DECODABLE_ACCEPT_ENCODING,
    accept_encoding_header,
    decodable_accept_encoding,
    httpx_accept_encoding,
)
from litellm.llms.openai.common_utils import BaseOpenAILLM


@pytest.fixture
def no_shared_sessions(monkeypatch):
    monkeypatch.setattr(litellm, "client_session", None)
    monkeypatch.setattr(litellm, "aclient_session", None)
    monkeypatch.setattr(litellm, "network_mock", False, raising=False)


def accept_encoding_of(client: httpx.Client | httpx.AsyncClient) -> str | None:
    return client.build_request("GET", "https://example.com").headers.get("Accept-Encoding")


@pytest.mark.parametrize(
    "advertised, expected",
    [
        ("gzip, deflate, br, zstd", "gzip, deflate, br"),
        ("gzip;q=1.0, deflate, br, zstd;q=0.9", "gzip;q=1.0, deflate, br"),
        (" zstd , gzip ", "gzip"),
        ("zstd", "identity"),
        ("gzip, deflate", "gzip, deflate"),
    ],
)
def test_zstd_is_the_only_encoding_dropped(advertised, expected):
    assert decodable_accept_encoding(advertised) == expected


def test_the_shipped_header_never_offers_zstd_but_still_offers_gzip():
    assert "zstd" not in DECODABLE_ACCEPT_ENCODING
    assert "gzip" in DECODABLE_ACCEPT_ENCODING
    assert accept_encoding_header() == {"Accept-Encoding": DECODABLE_ACCEPT_ENCODING}


def test_the_header_can_be_pointed_back_at_zstd_by_env_var(monkeypatch):
    monkeypatch.setenv(ACCEPT_ENCODING_OVERRIDE_ENV_VAR, "gzip, zstd")

    assert accept_encoding_header() == {"Accept-Encoding": "gzip, zstd"}


@pytest.mark.parametrize("override", ["", "   "])
def test_a_blank_override_leaves_the_default_in_place(monkeypatch, override):
    monkeypatch.setenv(ACCEPT_ENCODING_OVERRIDE_ENV_VAR, override)

    assert accept_encoding_header() == {"Accept-Encoding": DECODABLE_ACCEPT_ENCODING}


def test_an_httpx_release_that_stops_exposing_its_own_value_falls_back_to_gzip_and_deflate(monkeypatch):
    monkeypatch.delattr(httpx._client, "ACCEPT_ENCODING")

    assert httpx_accept_encoding() == "gzip, deflate"
    assert decodable_accept_encoding(httpx_accept_encoding()) == "gzip, deflate"


def test_legacy_http_handler_default_headers_drop_zstd():
    from litellm.llms.custom_httpx.httpx_handler import get_default_headers

    assert get_default_headers()["Accept-Encoding"] == DECODABLE_ACCEPT_ENCODING


@pytest.mark.parametrize(
    "build_client",
    [
        pytest.param(lambda: BaseLLM().create_client_session(), id="base_llm_session"),
        pytest.param(lambda: BaseOpenAILLM._get_sync_http_client(), id="openai_sdk_client"),
    ],
)
def test_sync_client_factories_ask_only_for_encodings_httpx_can_stream(build_client, no_shared_sessions):
    client = build_client()
    try:
        assert accept_encoding_of(client) == DECODABLE_ACCEPT_ENCODING
    finally:
        client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "build_client",
    [
        pytest.param(lambda: BaseLLM().create_aclient_session(), id="base_llm_session"),
        pytest.param(lambda: BaseOpenAILLM._get_async_http_client(), id="openai_sdk_client"),
        pytest.param(
            lambda: MCPClient(server_url="https://mcp.invalid/mcp")._create_httpx_client_factory()(),
            id="mcp_client",
        ),
    ],
)
async def test_async_client_factories_ask_only_for_encodings_httpx_can_stream(build_client, no_shared_sessions):
    client = build_client()
    try:
        assert accept_encoding_of(client) == DECODABLE_ACCEPT_ENCODING
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_mcp_client_keeps_the_headers_its_caller_passes():
    factory = MCPClient(server_url="https://mcp.invalid/mcp")._create_httpx_client_factory()

    client = factory(headers={"Authorization": "Bearer token", "Accept-Encoding": "gzip"})
    try:
        request = client.build_request("GET", "https://mcp.invalid/mcp")
        assert request.headers.get("Authorization") == "Bearer token"
        assert request.headers.get("Accept-Encoding") == "gzip"
    finally:
        await client.aclose()


@pytest.fixture
def frame_per_event_completions_upstream():
    """A completions upstream that answers each SSE event as its own zstd frame whenever the client offers zstd."""
    zstandard = pytest.importorskip("zstandard")

    events = tuple(
        f"data: {json.dumps(payload)}\n\n".encode()
        for payload in (
            {
                "id": "cmpl-1",
                "object": "text_completion",
                "created": 1,
                "model": "gpt-3.5-turbo-instruct",
                "choices": [{"text": "he", "index": 0, "finish_reason": None}],
            },
            {
                "id": "cmpl-1",
                "object": "text_completion",
                "created": 1,
                "model": "gpt-3.5-turbo-instruct",
                "choices": [{"text": "llo", "index": 0, "finish_reason": "stop"}],
            },
        )
    ) + (b"data: [DONE]\n\n",)
    seen_accept_encoding: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            accept_encoding = self.headers.get("Accept-Encoding", "")
            seen_accept_encoding["value"] = accept_encoding
            offers_zstd = "zstd" in accept_encoding
            frames = (
                tuple(zstandard.ZstdCompressor().compress(event) for event in events) if offers_zstd else events
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            if offers_zstd:
                self.send_header("Content-Encoding", "zstd")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for frame in frames:
                self.wfile.write(b"%X\r\n%s\r\n" % (len(frame), frame))
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        def log_message(self, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", seen_accept_encoding
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_streamed_text_completions_survive_an_upstream_that_frames_each_event(
    frame_per_event_completions_upstream, no_shared_sessions
):
    api_base, seen_accept_encoding = frame_per_event_completions_upstream

    stream = litellm.text_completion(
        model="text-completion-openai/gpt-3.5-turbo-instruct",
        prompt="say hello",
        stream=True,
        api_base=api_base,
        api_key="sk-not-a-real-key",
    )

    assert "".join(chunk.choices[0]["text"] or "" for chunk in stream) == "hello"
    assert "zstd" not in seen_accept_encoding["value"]


@pytest.mark.asyncio
async def test_async_streamed_text_completions_survive_an_upstream_that_frames_each_event(
    frame_per_event_completions_upstream, no_shared_sessions
):
    api_base, seen_accept_encoding = frame_per_event_completions_upstream

    stream = await litellm.atext_completion(
        model="text-completion-openai/gpt-3.5-turbo-instruct",
        prompt="say hello",
        stream=True,
        api_base=api_base,
        api_key="sk-not-a-real-key",
    )

    assert "".join([chunk.choices[0]["text"] or "" async for chunk in stream]) == "hello"
    assert "zstd" not in seen_accept_encoding["value"]
