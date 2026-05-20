"""
In-repo replacement for the Railway-hosted exampleopenaiendpoint that LiteLLM's
CI used to depend on. Returns static OpenAI-format responses so tests can run
without an external dependency.

Run standalone:
    python -m tests.mock_endpoints.openai_mock_server --host 0.0.0.0 --port 8090

The server is intentionally implemented with the Python standard library only
(no FastAPI / httpx / pydantic) so CI jobs can start it before installing test
dependencies.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger("openai_mock_server")

_EMBEDDING_DIM = 1536


def _now() -> int:
    return int(time.time())


def _chat_completion_body(
    model: str, content: str = "Hi! This is a mock response."
) -> dict:
    return {
        "id": "chatcmpl-mock-0001",
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "system_fingerprint": "fp_mock",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
        },
    }


def _text_completion_body(model: str) -> dict:
    return {
        "id": "cmpl-mock-0001",
        "object": "text_completion",
        "created": _now(),
        "model": model,
        "choices": [
            {
                "text": "Mock completion response.",
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }


def _embedding_body(model: str, input_count: int) -> dict:
    return {
        "object": "list",
        "model": model,
        "data": [
            {
                "object": "embedding",
                "index": i,
                "embedding": [0.0] * _EMBEDDING_DIM,
            }
            for i in range(max(input_count, 1))
        ],
        "usage": {"prompt_tokens": input_count or 1, "total_tokens": input_count or 1},
    }


def _rerank_body(query_id: str = "rerank-mock-0001") -> dict:
    return {
        "id": query_id,
        "results": [{"index": 0, "relevance_score": 0.99}],
        "meta": {
            "api_version": {"version": "1"},
            "billed_units": {"search_units": 1},
        },
    }


def _anthropic_message_body(model: str) -> dict:
    return {
        "id": "msg_mock_0001",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "Mock Anthropic message."}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }


def _chat_stream_chunks(model: str) -> list[dict]:
    base = {
        "id": "chatcmpl-mock-0001",
        "object": "chat.completion.chunk",
        "created": _now(),
        "model": model,
    }
    return [
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hi"},
                    "finish_reason": None,
                }
            ],
        },
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "! This is a mock"},
                    "finish_reason": None,
                }
            ],
        },
        {
            **base,
            "choices": [
                {"index": 0, "delta": {"content": " response."}, "finish_reason": None}
            ],
        },
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
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
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _route(self) -> str:
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
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
                            "created": _now(),
                            "owned_by": "mock",
                        },
                        {
                            "id": "gpt-4",
                            "object": "model",
                            "created": _now(),
                            "owned_by": "mock",
                        },
                    ],
                },
            )
            return
        self._send_json(200, {"status": "ok", "path": self.path})

    def do_POST(self) -> None:
        body = self._read_body()
        route = self._route()
        model = body.get("model") or "gpt-3.5-turbo"

        if route == "/chat/completions":
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
