"""Check installed SDK HTTP behavior against a recording loopback upstream."""

import asyncio
import importlib.util
import json
import os
import struct
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue
from threading import Event, Thread
from typing import Final
from unittest.mock import patch

ENVIRONMENT: Final = {"LITELLM_LOCAL_MODEL_COST_MAP": "True", "PYTHON_DOTENV_DISABLED": "1"}

with patch.dict(os.environ, ENVIRONMENT):
    import litellm

RESPONSES: Final[Queue[tuple[int, bytes]]] = Queue()
REQUESTS: Final[Queue[tuple[str, dict[str, str], bytes]]] = Queue()
ARRIVED: Final = Event()
RELEASE: Final = Event()
MESSAGES: Final = [{"role": "user", "content": "ping"}]
CHAT: Final = {
    "id": "chat-http-check",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
}


class RecordingUpstream(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_POST(self) -> None:
        REQUESTS.put((self.path, dict(self.headers), self.rfile.read(int(self.headers["Content-Length"]))))
        status, body = RESPONSES.get(timeout=10)
        ARRIVED.set()
        if status == 0:
            RELEASE.wait(timeout=10)
            return
        self.send_response(status)
        self.send_header("Content-Type", "text/event-stream" if body.startswith(b"data:") else "application/vnd.amazon.eventstream" if body[:1] == b"\x00" else "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def enqueue(body: object, status: int = 200) -> None:
    RESPONSES.put((status, json.dumps(body).encode()))


def enqueue_stream() -> None:
    chunk: Final = {**CHAT, "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "pong"}}]}
    RESPONSES.put((200, f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()))


def bedrock_event(event: str, payload: object) -> bytes:
    def encode_header(name: str, value: str) -> bytes:
        return bytes([len(name)]) + name.encode() + b"\x07" + struct.pack(">H", len(value)) + value.encode()

    headers: Final = encode_header(":message-type", "event") + encode_header(":event-type", event)
    content: Final = json.dumps(payload).encode()
    prelude: Final = struct.pack(">II", len(headers) + len(content) + 16, len(headers))
    frame: Final = prelude + struct.pack(">I", zlib.crc32(prelude)) + headers + content
    return frame + struct.pack(">I", zlib.crc32(frame))


def check_http(base: str) -> None:
    arguments: Final = dict(model="openai/gpt-4o", messages=MESSAGES, api_key="test-key", api_base=base + "/v1")
    enqueue(CHAT)
    response: Final = litellm.completion(**arguments)
    assert response.choices[0].message.content == "pong"
    assert response.usage.total_tokens == 5
    path, headers, body = REQUESTS.get(timeout=10)
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer test-key"
    assert json.loads(body)["messages"] == MESSAGES
    enqueue_stream()
    assert "".join(part.choices[0].delta.content or "" for part in litellm.completion(**arguments, stream=True)) == "pong"
    REQUESTS.get(timeout=10)
    for status, exception in ((401, litellm.AuthenticationError), (429, litellm.RateLimitError), (500, litellm.InternalServerError)):
        enqueue({"error": {"message": "controlled upstream failure"}}, status)
        try:
            litellm.completion(**arguments, num_retries=0, max_retries=0)
        except exception:
            REQUESTS.get(timeout=10)
        else:
            raise AssertionError(f"HTTP {status} did not raise {exception.__name__}")
    enqueue({"error": {"message": "retry once"}}, 429)
    enqueue(CHAT)
    assert litellm.completion(**arguments, num_retries=1).choices[0].message.content == "pong"
    REQUESTS.get(timeout=10)
    REQUESTS.get(timeout=10)

    async def check_async() -> None:
        enqueue(CHAT)
        result: Final = await litellm.acompletion(**arguments)
        assert result.choices[0].message.content == "pong"
        REQUESTS.get(timeout=10)
        enqueue_stream()
        stream: Final = await litellm.acompletion(**arguments, stream=True)
        assert "".join([part.choices[0].delta.content or "" async for part in stream]) == "pong"
        REQUESTS.get(timeout=10)
        ARRIVED.clear()
        RESPONSES.put((0, b""))
        task: Final = asyncio.create_task(litellm.acompletion(**arguments, num_retries=0, max_retries=0))
        assert await asyncio.to_thread(ARRIVED.wait, 10), "request did not reach upstream"
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            REQUESTS.get(timeout=10)
        else:
            raise AssertionError("cancellation did not propagate")
        finally:
            RELEASE.set()

    asyncio.run(check_async())
    enqueue({"object": "list", "model": "text-embedding-3-small",
             "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
             "usage": {"prompt_tokens": 3, "total_tokens": 3}})
    embedding: Final = litellm.embedding(model="text-embedding-3-small", input=["ping"], api_key="test-key", api_base=base + "/v1")
    assert embedding.data[0]["embedding"] == [0.1, 0.2]
    assert REQUESTS.get(timeout=10)[0] == "/v1/embeddings"
    RELEASE.clear()
    RESPONSES.put((0, b""))
    try:
        litellm.completion(**arguments, timeout=0.1, num_retries=0, max_retries=0)
    except litellm.Timeout:
        REQUESTS.get(timeout=10)
    else:
        raise AssertionError("stalled upstream did not time out")
    finally:
        RELEASE.set()
    for signed in ((False, True) if importlib.util.find_spec("boto3") else (False,)):
        for asynchronous in (False, True):
            enqueue({
                "output": {"message": {"role": "assistant", "content": [{"text": "pong"}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5},
                "metrics": {"latencyMs": 1},
            })
            bedrock: Final = dict(
                model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0", messages=MESSAGES,
                aws_region_name="us-east-1", aws_bedrock_runtime_endpoint=base,
                **({"aws_access_key_id": "test-key", "aws_secret_access_key": "test-secret", "api_key": ""}
                   if signed else {"api_key": "bearer-key"}),
            )
            result: Final = asyncio.run(litellm.acompletion(**bedrock)) if asynchronous else litellm.completion(**bedrock)
            assert result.choices[0].message.content == "pong"
            assert result.usage.total_tokens == 5
            path, headers, body = REQUESTS.get(timeout=10)
            assert path.endswith("/converse")
            assert {key.lower(): value for key, value in headers.items()}["authorization"].startswith(
                "AWS4-HMAC-SHA256" if signed else "Bearer bearer-key"
            )
            assert json.loads(body)["messages"][0]["content"][0]["text"] == "ping"
    if importlib.util.find_spec("botocore") is not None:
        RESPONSES.put((200, b"".join((
            bedrock_event("messageStart", {"role": "assistant"}),
            bedrock_event("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "pong"}}),
            bedrock_event("messageStop", {"stopReason": "end_turn"}),
            bedrock_event("metadata", {"usage": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5}}),
        ))))
        streamed: Final = litellm.completion(
            model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0", messages=MESSAGES,
            api_key="bearer-key", aws_region_name="us-east-1", aws_bedrock_runtime_endpoint=base, stream=True,
        )
        assert "".join(part.choices[0].delta.content or "" for part in streamed) == "pong"
        assert REQUESTS.get(timeout=10)[0].endswith("/converse-stream")
    enqueue({"id": "msg-http-check", "type": "message", "role": "assistant", "model": "claude-3-sonnet-20240229",
             "content": [{"type": "text", "text": "pong"}], "stop_reason": "end_turn", "stop_sequence": None,
             "usage": {"input_tokens": 3, "output_tokens": 2}})
    anthropic: Final = litellm.completion(model="anthropic/claude-3-sonnet-20240229", messages=MESSAGES,
                                        api_key="test-key", api_base=base, max_tokens=16)
    assert anthropic.choices[0].message.content == "pong"
    assert anthropic.usage.total_tokens == 5
    assert REQUESTS.get(timeout=10)[0] == "/v1/messages"
    print("PASS installed HTTP: sync/async, streaming, usage, error mapping, retries, cancellation, Bedrock auth")


def main() -> None:
    with ThreadingHTTPServer(("127.0.0.1", 0), RecordingUpstream) as server:
        worker: Final = Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            check_http(f"http://127.0.0.1:{server.server_port}")
            assert RESPONSES.empty() and REQUESTS.empty(), "unconsumed HTTP exchanges"
        finally:
            RELEASE.set()
            server.shutdown()
            worker.join(timeout=10)


if __name__ == "__main__":
    with patch.dict(os.environ, ENVIRONMENT):
        main()
