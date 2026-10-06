"""Exercise an installed SDK against a local HTTP server, without test dependencies."""

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

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
os.environ["PYTHON_DOTENV_DISABLED"] = "1"

import litellm
from litellm._version import get_distribution_name

REPLIES: Final[Queue[tuple[int, bytes, str]]] = Queue()
REQUESTS: Final[Queue[tuple[str, dict[str, str], bytes]]] = Queue()
ARRIVED: Final = Event()
RELEASE: Final = Event()


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_POST(self) -> None:
        REQUESTS.put((self.path, dict(self.headers.items()), self.rfile.read(int(self.headers["Content-Length"]))))
        status, body, media_type = REPLIES.get(timeout=5)
        ARRIVED.set()
        if status == 0:
            RELEASE.wait(timeout=5)
            return
        self.send_response(status)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def reply(body: object, status: int = 200) -> None:
    REPLIES.put((status, json.dumps(body).encode(), "application/json"))


CHAT: Final = {
    "id": "chat-test",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-6.1-sol",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
}
MESSAGES: Final = [{"role": "user", "content": "ping"}]


def stream_reply() -> None:
    chunk: Final = {
        "id": "chat-test",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-6.1-sol",
        "choices": [{"index": 0, "delta": {"content": "pong"}, "finish_reason": None}],
    }
    REPLIES.put((200, ("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode(), "text/event-stream"))


def bedrock_event(name: str, payload: object) -> bytes:
    def header(key: str, value: str) -> bytes:
        return bytes([len(key)]) + key.encode() + b"\x07" + struct.pack(">H", len(value)) + value.encode()

    headers: Final = (
        header(":message-type", "event") + header(":event-type", name) + header(":content-type", "application/json")
    )
    body: Final = json.dumps(payload).encode()
    prelude: Final = struct.pack(">II", 16 + len(headers) + len(body), len(headers))
    message: Final = prelude + struct.pack(">I", zlib.crc32(prelude)) + headers + body
    return message + struct.pack(">I", zlib.crc32(message))


def check_http(base: str) -> None:
    kwargs: Final = dict(model="openai/gpt-6.1-sol", messages=MESSAGES, api_base=base + "/v1", api_key="test-key")
    reply(CHAT)
    response: Final = litellm.completion(**kwargs)
    assert response.choices[0].message.content == "pong"
    assert response.usage.total_tokens == 5
    assert litellm.completion_cost(completion_response=response) >= 0
    path, headers, body = REQUESTS.get(timeout=5)
    assert path == "/v1/chat/completions" and json.loads(body)["messages"] == MESSAGES
    assert headers["Authorization"] == "Bearer test-key"
    stream_reply()
    assert "".join(c.choices[0].delta.content or "" for c in litellm.completion(**kwargs, stream=True)) == "pong"
    REQUESTS.get(timeout=5)
    reply(
        {
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        }
    )
    embedding: Final = litellm.embedding(
        model="text-embedding-3-small", input=["ping"], api_base=base + "/v1", api_key="test-key"
    )
    assert embedding.data[0]["embedding"] == [0.1, 0.2]
    REQUESTS.get(timeout=5)
    for status, error_type in (
        (401, litellm.AuthenticationError),
        (429, litellm.RateLimitError),
        (500, litellm.InternalServerError),
    ):
        reply({"error": {"message": "upstream failure", "type": "test_error"}}, status)
        try:
            litellm.completion(**kwargs, max_retries=0, num_retries=0)
        except error_type:
            pass
        else:
            raise AssertionError(f"HTTP {status} did not surface its SDK exception")
        REQUESTS.get(timeout=5)
    reply({"error": {"message": "transient", "type": "rate_limit_error"}}, 429)
    reply(CHAT)
    retried: Final = litellm.completion(**kwargs, num_retries=1)
    assert retried.choices[0].message.content == "pong"
    REQUESTS.get(timeout=5)
    REQUESTS.get(timeout=5)
    REPLIES.put((0, b"", "application/json"))
    try:
        litellm.completion(**kwargs, timeout=0.1, max_retries=0, num_retries=0)
    except litellm.Timeout:
        pass
    else:
        raise AssertionError("stalled upstream did not time out")
    RELEASE.set()
    REQUESTS.get(timeout=5)
    print("sync completion, streaming, embedding, usage, errors, retry and timeout passed")

    async def asynchronous() -> None:
        reply(CHAT)
        result: Final = await litellm.acompletion(**kwargs)
        assert result.choices[0].message.content == "pong"
        REQUESTS.get(timeout=5)
        stream_reply()
        stream: Final = await litellm.acompletion(**kwargs, stream=True)
        parts: Final = [c.choices[0].delta.content or "" async for c in stream]
        assert "".join(parts) == "pong"
        REQUESTS.get(timeout=5)
        ARRIVED.clear()
        RELEASE.clear()
        REPLIES.put((0, b"", "application/json"))
        task: Final = asyncio.create_task(litellm.acompletion(**kwargs, timeout=10, max_retries=0, num_retries=0))
        assert await asyncio.to_thread(ARRIVED.wait, 5), "async request never reached upstream"
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("caller cancellation was swallowed")
        finally:
            RELEASE.set()
        REQUESTS.get(timeout=5)

    asyncio.run(asynchronous())
    print("async completion, streaming and cancellation passed")

    for signed in (False, True) if importlib.util.find_spec("boto3") else (False,):
        for asynchronous_call in (False, True):
            reply(
                {
                    "output": {"message": {"role": "assistant", "content": [{"text": "pong"}]}},
                    "stopReason": "end_turn",
                    "usage": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5},
                    "metrics": {"latencyMs": 1},
                }
            )
            bedrock_params: Final = dict(
                model="bedrock/anthropic.claude-sonnet-5-5",
                messages=MESSAGES,
                aws_region_name="us-east-1",
                aws_bedrock_runtime_endpoint=base,
                **(
                    {"api_key": "", "aws_access_key_id": "test-key", "aws_secret_access_key": "test-secret"}
                    if signed
                    else {"api_key": "bearer-token"}
                ),
            )
            result: Final = (
                asyncio.run(litellm.acompletion(**bedrock_params))
                if asynchronous_call
                else litellm.completion(**bedrock_params)
            )
            assert result.choices[0].message.content == "pong" and result.usage.total_tokens == 5
            path, headers, body = REQUESTS.get(timeout=5)
            normalized: Final = {k.lower(): v for k, v in headers.items()}
            assert path.endswith("/converse")
            assert normalized["authorization"].startswith("AWS4-HMAC-SHA256" if signed else "Bearer bearer-token")
            assert json.loads(body)["messages"][0]["content"][0]["text"] == "ping"
    events: Final = b"".join(
        (
            bedrock_event("messageStart", {"role": "assistant"}),
            bedrock_event("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "pong"}}),
            bedrock_event("messageStop", {"stopReason": "end_turn"}),
            bedrock_event("metadata", {"usage": {"inputTokens": 3, "outputTokens": 2, "totalTokens": 5}}),
        )
    )
    REPLIES.put((200, events, "application/vnd.amazon.eventstream"))
    try:
        streamed: Final = litellm.completion(
            model="bedrock/anthropic.claude-sonnet-5-5",
            messages=MESSAGES,
            api_key="bearer-token",
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=base,
            stream=True,
        )
        content: Final = "".join(c.choices[0].delta.content or "" for c in streamed)
    except (ImportError, litellm.APIConnectionError) as error:
        if importlib.util.find_spec("botocore") is not None or f"{get_distribution_name()}[aws]" not in str(error):
            raise AssertionError("unexpected Bedrock streaming failure") from error
    else:
        assert content == "pong"
    REQUESTS.get(timeout=5)
    print("Bedrock bearer, installed AWS signing, streaming and missing-extra guidance passed")
    reply(
        {
            "id": "msg-test",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5-5",
            "content": [{"type": "text", "text": "pong"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }
    )
    anthropic: Final = litellm.completion(
        model="anthropic/claude-sonnet-5-5", messages=MESSAGES, api_key="test-key", api_base=base, max_tokens=16
    )
    assert anthropic.choices[0].message.content == "pong" and anthropic.usage.total_tokens == 5
    path, headers, body = REQUESTS.get(timeout=5)
    assert path == "/v1/messages"
    assert {k.lower(): v for k, v in headers.items()}["x-api-key"] == "test-key"
    print("Anthropic request translation and usage passed")


def check_search_http(base: str) -> None:
    result: Final = {"title": "Search result", "url": "https://example.com", "snippet": "Retained search", "date": "2026-01-02"}
    reply({"results": [result]})
    response: Final = litellm.search(query="hello", search_provider="perplexity", api_key="test-search-key", api_base=base)
    assert response.results[0].url == result["url"] and response.results[0].date == result["date"]
    path, headers, body = REQUESTS.get(timeout=5)
    assert json.loads(body)["query"] == "hello"
    assert {k.lower(): v for k, v in headers.items()}["authorization"] == "Bearer test-search-key"

    async def search_async() -> None:
        reply({"results": [result]})
        response: Final = await litellm.asearch(query="hello", search_provider="perplexity", api_key="test-search-key", api_base=base)
        assert response.results[0].snippet == result["snippet"]
        path, headers, body = REQUESTS.get(timeout=5)
        assert json.loads(body)["query"] == "hello"

    asyncio.run(search_async())
    print("sync/async Perplexity search works independently of Brave date parsing")


if __name__ == "__main__":
    with ThreadingHTTPServer(("127.0.0.1", 0), Upstream) as server:
        thread: Final = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            check_http(f"http://127.0.0.1:{server.server_port}")
            check_search_http(f"http://127.0.0.1:{server.server_port}")
            assert REPLIES.empty() and REQUESTS.empty(), "unconsumed HTTP exchanges"
        finally:
            RELEASE.set()
            server.shutdown()
            thread.join(timeout=5)
