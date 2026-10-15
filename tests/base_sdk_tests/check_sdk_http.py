"""Check installed SDK HTTP behavior against a recording loopback upstream."""

import asyncio
import importlib.util
import json
import os
import struct
import zlib
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping, Sequence
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Queue
from threading import Event, Thread
from typing import Final, cast
from unittest import SkipTest, skipIf
from unittest.mock import patch

from pydantic import BaseModel, JsonValue, TypeAdapter

ENVIRONMENT: Final = {"LITELLM_LOCAL_MODEL_COST_MAP": "True", "PYTHON_DOTENV_DISABLED": "1"}

with patch.dict(os.environ, ENVIRONMENT):
    import litellm

RESPONSES: Final[Queue[tuple[int, bytes]]] = Queue()
REQUESTS: Final[Queue[tuple[str, dict[str, str], bytes]]] = Queue()
ARRIVED: Final = Event()
RELEASE: Final = Event()
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
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
        headers: Final = {name.lower(): value for name, value in self.headers.items()}
        REQUESTS.put((self.path, headers, self.rfile.read(int(self.headers["Content-Length"]))))
        status, body = RESPONSES.get(timeout=10)
        ARRIVED.set()
        if status == 0:
            RELEASE.wait(timeout=10)
            return
        self.send_response(status)
        self.send_header(
            "Content-Type",
            "text/event-stream"
            if body.startswith((b"data:", b"event:"))
            else "application/vnd.amazon.eventstream"
            if body[:1] == b"\x00"
            else "application/json",
        )
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


def payload(response: object) -> Mapping[str, JsonValue]:
    assert isinstance(response, BaseModel), response
    return JSON_OBJECT.validate_json(response.model_dump_json(exclude_none=True))


@skipIf(importlib.util.find_spec("mcp") is None, "BUG: LIT-9333 completion(tools=...) imports optional mcp")
def check_tools(base: str, asynchronous: bool = False) -> None:
    tools: Final = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    call: Final = {
        "id": "call-http-check",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"city":"Paris"}'},
    }
    arguments: Final = dict(model="openai/gpt-6", messages=MESSAGES, api_key="test-key", api_base=base + "/v1")
    choices: Final = [
        {"index": 0, "message": {"role": "assistant", "tool_calls": [call]}, "finish_reason": "tool_calls"}
    ]
    enqueue({**CHAT, "choices": choices})
    result: Final = (
        asyncio.run(litellm.acompletion(**arguments, tools=tools))
        if asynchronous
        else litellm.completion(**arguments, tools=tools)
    )
    assert payload(result)["choices"] == [{**choices[0], "provider_specific_fields": {}}]
    path, headers, body = REQUESTS.get(timeout=10)
    assert path == "/v1/chat/completions"
    assert headers["authorization"] == "Bearer test-key"
    assert json.loads(body) == {"model": "gpt-6", "messages": MESSAGES, "tools": tools}


@skipIf(importlib.util.find_spec("mcp") is None, "BUG: LIT-9333 responses() imports optional mcp")
def check_responses(base: str, asynchronous: bool = False) -> None:
    output: Final[Sequence[JsonValue]] = [
        {
            "id": "msg-http-check",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "pong", "annotations": []}],
        }
    ]
    enqueue(
        {
            "id": "resp-http-check",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-6",
            "output": output,
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        }
    )
    arguments: Final = dict(model="openai/gpt-6", input="ping", api_key="test-key", api_base=base + "/v1")
    response: Final = asyncio.run(litellm.aresponses(**arguments)) if asynchronous else litellm.responses(**arguments)
    assert payload(response)["output"] == output
    assert payload(response)["usage"] == {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
    path, headers, body = REQUESTS.get(timeout=10)
    assert path == "/v1/responses"
    assert headers["authorization"] == "Bearer test-key"
    assert json.loads(body) == {"model": "gpt-6", "input": "ping"}


@skipIf(importlib.util.find_spec("fastapi") is None, "BUG: LIT-9333 cache hits import proxy-only fastapi")
def check_cache_hit(base: str) -> None:
    from litellm.caching.caching import Cache
    from litellm.types.caching import LiteLLMCacheType

    arguments: Final = dict(
        model="openai/gpt-6",
        messages=MESSAGES,
        api_key="test-key",
        api_base=base + "/v1",
        caching=True,
        timeout=3,
        num_retries=0,
        max_retries=0,
    )
    with patch.object(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL)):
        enqueue(CHAT)
        first: Final = litellm.completion(**arguments)
        assert REQUESTS.get(timeout=10)[0] == "/v1/chat/completions"
        cached: Final = litellm.completion(**arguments)
        assert payload(cached) == payload(first)
        assert payload(cached)["choices"] == [{
            "index": 0, "message": {"role": "assistant", "content": "pong"},
            "finish_reason": "stop", "provider_specific_fields": {},
        }]
        assert payload(cached)["usage"] == CHAT["usage"]
        assert REQUESTS.empty(), "the second SDK call missed the response cache"


@skipIf(importlib.util.find_spec("fastapi") is None, "BUG: LIT-9333 Messages streaming imports proxy-only fastapi")
def check_messages_stream_and_callback(base: str) -> None:
    from litellm.integrations.custom_logger import CustomLogger
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    successes: Final[Queue[Mapping[str, object]]] = Queue()
    failures: Final[Queue[object]] = Queue()

    class Probe(CustomLogger):
        async def async_log_success_event(
            self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
        ) -> None:
            if kwargs.get("litellm_call_id") == "base-sdk-messages-check":
                successes.put(kwargs)

        async def async_log_failure_event(
            self, kwargs: Mapping[str, object], response_obj: object, start_time: object, end_time: object
        ) -> None:
            if kwargs.get("litellm_call_id") == "base-sdk-messages-check":
                failures.put(response_obj)

    frames: Final[tuple[Mapping[str, JsonValue], ...]] = (
        {
            "type": "message_start",
            "message": {
                "id": "msg-stream-http-check",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5-5",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "pong"}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 2},
        },
        {"type": "message_stop"},
    )
    RESPONSES.put(
        (200, b"".join(f"event: {frame['type']}\ndata: {json.dumps(frame)}\n\n".encode() for frame in frames))
    )

    async def check_async() -> None:
        with patch.object(litellm, "callbacks", [Probe()]):
            create: Final = cast(  # cast-ok: SDK facade exposes untyped kwargs and iterator items
                Callable[..., Awaitable[object]], litellm.anthropic.messages.acreate
            )
            stream: Final = await create(
                model="anthropic/claude-sonnet-5-5",
                messages=MESSAGES,
                api_key="test-key",
                api_base=base,
                max_tokens=16,
                stream=True,
                litellm_call_id="base-sdk-messages-check",
            )
            assert isinstance(stream, AsyncIterable), stream
            iterator: Final = cast(AsyncIterable[object], stream)  # cast-ok: SDK returns an unparameterized iterator
            chunks: Final = tuple([frame async for frame in iterator])
            assert all(isinstance(frame, bytes) for frame in chunks), chunks
            received: Final = b"".join(frame for frame in chunks if isinstance(frame, bytes))
            assert tuple(json.loads(line[6:]) for line in received.splitlines() if line.startswith(b"data: ")) == frames
            success: Final = await asyncio.to_thread(successes.get, True, 10)
            await GLOBAL_LOGGING_WORKER.flush()
            logging: Final = success["standard_logging_object"]
            assert isinstance(logging, dict)
            assert (logging["prompt_tokens"], logging["completion_tokens"], logging["total_tokens"]) == (3, 2, 5)
            assert successes.empty(), "a successful stream fired more than one success callback"
            assert failures.empty(), "a successful stream fired a failure callback"

    asyncio.run(check_async())
    path, headers, body = REQUESTS.get(timeout=10)
    assert path == "/v1/messages"
    assert headers["x-api-key"] == "test-key"
    assert json.loads(body) == {"model": "claude-sonnet-5-5", "messages": MESSAGES, "max_tokens": 16, "stream": True}


def check_http(base: str) -> None:
    arguments: Final = dict(model="openai/gpt-4o", messages=MESSAGES, api_key="test-key", api_base=base + "/v1")
    enqueue(CHAT)
    response: Final = litellm.completion(**arguments)
    assert response.choices[0].message.content == "pong"
    assert response.usage.total_tokens == 5
    path, headers, body = REQUESTS.get(timeout=10)
    assert path == "/v1/chat/completions"
    assert headers["authorization"] == "Bearer test-key"
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
            assert headers["authorization"].startswith(
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
            for check in (
                check_tools,
                partial(check_tools, asynchronous=True),
                check_responses,
                partial(check_responses, asynchronous=True),
                check_cache_hit,
                check_messages_stream_and_callback,
            ):
                try:
                    check(f"http://127.0.0.1:{server.server_port}")
                except SkipTest as skipped:
                    print(f"SKIP: {skipped}")
            assert RESPONSES.empty() and REQUESTS.empty(), "unconsumed HTTP exchanges"
        finally:
            RELEASE.set()
            server.shutdown()
            worker.join(timeout=10)


if __name__ == "__main__":
    with patch.dict(os.environ, ENVIRONMENT):
        main()
