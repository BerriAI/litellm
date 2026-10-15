import json
import os
import subprocess
import sys
import threading
import uuid
from collections.abc import Mapping
from typing import Final

from integration._support.client import eventually
from integration._support.wire import Reply, Request, wire_server
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

CALLS: Final = {
    "sync": "completion",
    "sync_stream": "completion",
    "async": "acompletion",
    "async_stream": "acompletion",
}
SDK_CALLS: Final = """
import asyncio
import sys

import litellm

litellm.callbacks = ["langfuse_otel"]
api_base, marker = sys.argv[1], sys.argv[2]


def request(call):
    return {
        "model": "openai/gpt-4o-mini",
        "api_base": api_base,
        "api_key": "synthetic-provider-key",
        "messages": [{"role": "user", "content": f"{marker}-{call}"}],
        "stream": call.endswith("_stream"),
    }


async def async_calls():
    await litellm.acompletion(**request("async"))
    async for _ in await litellm.acompletion(**request("async_stream")):
        pass


litellm.completion(**request("sync"))
for _ in litellm.completion(**request("sync_stream")):
    pass
asyncio.run(async_calls())
print("calls done", flush=True)
sys.stdin.read()
"""


def _chat_reply(request: Request) -> Reply:
    body: Final = json.loads(request.body)
    content: Final = body["messages"][0]["content"]
    if body.get("stream"):
        delta: Final = {"id": content, "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini"}
        return Reply(
            content_type="text/event-stream",
            chunks=(
                f"data: {json.dumps({**delta, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': 'ok'}}]})}\n\n".encode(),
                f"data: {json.dumps({**delta, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n".encode(),
                b"data: [DONE]\n\n",
            ),
        )
    return Reply(
        body=json.dumps(
            {
                "id": content,
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            }
        ).encode()
    )


class GenerationSink:
    def __init__(self) -> None:
        self.lock: Final = threading.Lock()
        self.generations: Final[list[Mapping[str, str]]] = []  # mutable-ok: spans accumulate as the exporter delivers

    def respond(self, request: Request) -> Reply:
        if not request.target.endswith("/v1/traces"):
            return Reply(body=b"")
        spans: Final = tuple(
            {pair.key: pair.value.string_value for pair in span.attributes}
            for resource in ExportTraceServiceRequest.FromString(request.body).resource_spans
            for scope in resource.scope_spans
            for span in scope.spans
        )
        with self.lock:
            self.generations.extend(span for span in spans if span.get("langfuse.observation.type") == "generation")
        return Reply(body=b"")

    def by_call(self, marker: str) -> dict[str, tuple[Mapping[str, str], ...]]:
        with self.lock:
            snapshot: Final = tuple(self.generations)
        return {
            call: tuple(span for span in snapshot if f'"{marker}-{call}"' in span.get("langfuse.observation.input", ""))
            for call in CALLS
        }


def test_sdk_sync_and_async_calls_derive_trace_name_and_io_from_their_call_type() -> None:
    marker: Final = "lit8281-sdk-" + uuid.uuid4().hex
    sink: Final = GenerationSink()
    with wire_server(sink.respond) as destination, wire_server(_chat_reply) as provider:
        child: Final = subprocess.Popen(
            [sys.executable, "-P", "-c", SDK_CALLS, provider.url + "/v1", marker],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "LANGFUSE_PUBLIC_KEY": "pk-lf-sdk",
                "LANGFUSE_SECRET_KEY": "sk-lf-sdk",
                "LANGFUSE_HOST": destination.url,
                "OTEL_BSP_SCHEDULE_DELAY": "200",
            },
        )
        try:
            spans: Final = eventually(
                lambda: sink.by_call(marker),
                lambda found: all(len(records) == 1 for records in found.values()) or child.poll() is not None,
                seconds=70,
            )
        finally:
            output: Final = child.communicate(input=b"", timeout=60)[0].decode()
    assert {call: len(records) for call, records in spans.items()} == dict.fromkeys(CALLS, 1), output
    for call, call_type in CALLS.items():
        attributes = spans[call][0]
        assert attributes["langfuse.trace.name"] == f"litellm-{call_type}", (call, attributes)
        assert attributes["langfuse.trace.input"] == attributes["langfuse.observation.input"], (call, attributes)
        assert attributes["langfuse.trace.output"] == attributes["langfuse.observation.output"], (call, attributes)
        assert json.loads(attributes["langfuse.trace.output"]) == {"role": "assistant", "content": "ok"}, call
