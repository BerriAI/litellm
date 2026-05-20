"""
In-repo replacement for the Railway-hosted exampleopenaiendpoint that LiteLLM's
CI used to depend on. Returns static OpenAI-format responses so tests can run
without an external dependency.

Run standalone:
    python -m tests.mock_endpoints.openai_mock_server --host 0.0.0.0 --port 8090

The server is intentionally implemented with the Python standard library only
(no FastAPI / httpx / pydantic) so CI jobs can start it before installing test
dependencies.

Response shapes are matched to the historical Railway endpoint to preserve
test intent:
  * /chat/completions          -> OpenAI chat.completion (non-stream) or SSE
                                  stream of chat.completion.chunk events.
                                  Content: "Hello this is a test response from
                                  a fixed OpenAI endpoint. " (stream) or
                                  "\\n\\nHello there, how may I assist you
                                  today?" (non-stream).
  * /completions               -> OpenAI text_completion.
  * /embeddings                -> OpenAI embedding list.
  * /rerank, /v2/rerank        -> Cohere-shape rerank result.
  * /v1/messages               -> Anthropic Messages.

Special model handling matches Railway:
  * model == "429" returns HTTP 429 (used by fake-azure-endpoint fixtures).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger("openai_mock_server")

_EMBEDDING_DIM = 1536
_CREATED_TS = 1677652288
_SYSTEM_FINGERPRINT = "fp_44709d6fcb"
_CHAT_CONTENT = "\n\nHello there, how may I assist you today?"
_STREAM_TOKENS = [
    "Hello ",
    "this ",
    "is ",
    "a ",
    "test ",
    "response ",
    "from ",
    "a ",
    "fixed ",
    "OpenAI ",
    "endpoint. ",
]


def _chat_completion_body(model: str) -> dict:
    return {
        "id": "chatcmpl-c055ccfa83b84490af310d4bb5552422",
        "object": "chat.completion",
        "created": _CREATED_TS,
        "model": model,
        "system_fingerprint": _SYSTEM_FINGERPRINT,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": _CHAT_CONTENT},
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
    }


def _text_completion_body(model: str) -> dict:
    return {
        "id": "cmpl-9B2ycsf0odECdLmrVzm2y8Q12csjW",
        "object": "text_completion",
        "created": _CREATED_TS,
        "model": model,
        "system_fingerprint": None,
        "choices": [
            {
                "text": "\n\nA test request, how intriguing\n"
                "An invitation for knowledge bringing\nWith words",
                "index": 0,
                "logprobs": None,
                "finish_reason": "length",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 16, "total_tokens": 26},
    }


def _embedding_body(model: str, input_count: int) -> dict:
    # Match Railway: a constant 1536-d vector tiled from a short pattern.
    pattern = [
        -0.006929283495992422,
        -0.005336422007530928,
        -4.547132266452536e-05,
        -0.024047505110502243,
    ]
    vector = (pattern * ((_EMBEDDING_DIM // len(pattern)) + 1))[:_EMBEDDING_DIM]
    return {
        "object": "list",
        "model": model,
        "data": [
            {"object": "embedding", "index": i, "embedding": vector}
            for i in range(max(input_count, 1))
        ],
        "usage": {"prompt_tokens": input_count or 1, "total_tokens": input_count or 1},
    }


def _rerank_body() -> dict:
    return {
        "id": "rerank-mock-0001",
        "results": [{"index": 0, "relevance_score": 0.99}],
        "meta": {
            "api_version": {"version": "1"},
            "billed_units": {"search_units": 1},
        },
    }


def _anthropic_message_body(model: str) -> dict:
    return {
        "id": "msg_01G7MsdWPT2JZMUuc1UXRavn",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": _CHAT_CONTENT}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
    }


def _chat_stream_chunks(model: str) -> list[dict]:
    base = {
        "id": "chatcmpl-b0e9cfbf26d148928e8501842b2af4de",
        "object": "chat.completion.chunk",
        "created": _CREATED_TS,
        "model": model,
    }
    # Match Railway: deltas carry only `content` (no role, no finish_reason),
    # and the stream ends after the last token without a [DONE] sentinel.
    return [
        {**base, "choices": [{"index": 0, "delta": {"content": token}}]}
        for token in _STREAM_TOKENS
    ]


class MockOpenAIHandler(BaseHTTPRequestHandler):
    server_version = "MockOpenAI/1.0"

    def log_message(
        self, format: str, *args
    ) -> None:  # noqa: A002 - signature dictated by base
        logger.debug(
            "%s - - [%s] %s",
            self.address_string(),
            self.log_date_time_string(),
            format % args,
        )

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return {}

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, chunks: list[dict]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
            self.wfile.flush()
        # OpenAI clients (httpx / openai-python) terminate on a `data: [DONE]`
        # sentinel; Railway omitted it and relied on connection close. Sending
        # it explicitly is safe (LiteLLM tolerates either) and avoids the
        # 30s-keep-alive hang seen with the raw stdlib server.
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    # Path prefixes that simulate latency before responding. Used by tests
    # like test_lowest_latency_routing_with_timeouts that historically pointed
    # the "slow deployment" at a separate Railway URL which was naturally slow.
    _DELAY_PREFIXES = {
        "/slow": 2.0,
    }

    def _maybe_delay(self) -> None:
        path = self.path.split("?", 1)[0]
        for prefix, seconds in self._DELAY_PREFIXES.items():
            if path.startswith(prefix + "/") or path == prefix:
                time.sleep(seconds)
                return

    def _stripped_path(self) -> str:
        """Path with `/slow` (delay) prefix removed but `/v1`, `/v2` preserved."""
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        for prefix in self._DELAY_PREFIXES:
            if path.startswith(prefix + "/"):
                return path[len(prefix) :]
            if path == prefix:
                return "/"
        return path

    def _route(self) -> str:
        path = self._stripped_path()
        for prefix in ("/v1", "/v2"):
            if path.startswith(prefix + "/"):
                return path[len(prefix) :]
            if path == prefix:
                return "/"
        return path

    def do_GET(self) -> None:
        route = self._route()
        if route == "/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "gpt-3.5-turbo",
                            "object": "model",
                            "created": _CREATED_TS,
                            "owned_by": "mock",
                        },
                        {
                            "id": "gpt-4",
                            "object": "model",
                            "created": _CREATED_TS,
                            "owned_by": "mock",
                        },
                    ],
                },
            )
            return
        self._send_json(200, {"status": "ok", "path": self.path})

    def do_POST(self) -> None:
        body = self._read_body()
        self._maybe_delay()
        raw_path = self._stripped_path()
        route = self._route()
        model = body.get("model") or "gpt-3.5-turbo"

        # Railway convention: `model == "429"` triggers a rate-limit response.
        # Used by `fake-azure-endpoint` fixtures to exercise retry/fallback paths.
        if model == "429" and route in ("/chat/completions", "/completions"):
            self._send_json(429, {"detail": "Too many requests"})
            return

        if route == "/chat/completions":
            # Railway quirk we have to preserve: POST /v1/chat/completions
            # returns an Anthropic-shape Messages response (vs. the OpenAI shape
            # served at /chat/completions). LiteLLM's OpenAI parser raises
            # InternalServerError on that response, and a few tests
            # (e.g. test_router_prompt_caching) defensively rely on that path
            # via try/except. Matching the quirk keeps their intent intact.
            if raw_path.startswith("/v1/"):
                self._send_json(200, _anthropic_message_body(model))
                return
            if body.get("stream"):
                self._send_sse(_chat_stream_chunks(model))
            else:
                self._send_json(200, _chat_completion_body(model))
            return

        if route == "/completions":
            self._send_json(200, _text_completion_body(model))
            return

        if route == "/embeddings":
            inputs = body.get("input")
            count = len(inputs) if isinstance(inputs, list) else 1
            self._send_json(200, _embedding_body(model, count))
            return

        if route == "/rerank":
            self._send_json(200, _rerank_body())
            return

        if route == "/messages":
            self._send_json(200, _anthropic_message_body(model))
            return

        # Catch-all: respond OK so tests asserting on api_base reachability pass.
        self._send_json(200, {"status": "ok", "path": self.path})


def serve(host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), MockOpenAIHandler)
    logger.info("mock openai server listening on http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Static OpenAI-format mock server for LiteLLM CI."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    serve(args.host, args.port)


if __name__ == "__main__":
    main()
