import asyncio
import json
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final

import anthropic
import openai
import yaml
from integration._support.client import Gateway, JsonValue, eventually, object_value
from integration._support.wire import Reply, Request

BUCKET: Final = "integration-bucket"
PREFIX: Final = "integration-logs"


@dataclass(slots=True)
class RecordingS3Sink:
    """Records every accepted PUT body by target, tracks peak concurrency, and can reject a leading
    run of PUT attempts with a chosen status before accepting. Serves stored bodies back on GET."""

    fail_attempts: int = 0
    fail_until: float = 0.0
    fail_status: int = 503
    fail_code: str = "SinkFailure"
    fail_body: bytes | None = None
    delay_seconds: float = 0.5
    lock: threading.Lock = field(default_factory=threading.Lock)
    in_flight: int = 0
    peak: int = 0
    attempts: int = 0
    attempt_log: list[tuple[float, int]] = field(default_factory=list)  # mutable-ok: appended under lock per PUT
    store: dict[str, bytes] = field(default_factory=dict)  # mutable-ok: GET reads must see writes from earlier PUTs

    def respond(self, request: Request) -> Reply:
        if request.method == "GET":
            body: Final = self.store.get(request.target)
            if body is None:
                return Reply(status=404)
            return Reply(body=body)
        assert request.method == "PUT", request.method
        assert request.target.startswith(f"/{BUCKET}/{PREFIX}/"), request.target
        with self.lock:
            self.attempts += 1
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            self.attempt_log.append((time.time(), self.in_flight))
            failing: Final = self.attempts <= self.fail_attempts or time.time() < self.fail_until
            if not failing:
                self.store[request.target] = request.body
        time.sleep(self.delay_seconds)
        with self.lock:
            self.in_flight -= 1
        if failing:
            return Reply(
                status=self.fail_status,
                body=self.fail_body
                if self.fail_body is not None
                else f"<Error><Code>{self.fail_code}</Code></Error>".encode(),
                content_type="application/xml",
            )
        return Reply()

    def peak_between(self, start: float, end: float) -> int:
        with self.lock:
            samples: Final = tuple(in_flight for when, in_flight in self.attempt_log if start <= when < end)
        return max(samples, default=0)

    def objects(self) -> Mapping[str, bytes]:
        with self.lock:
            return MappingProxyType(dict(self.store))

    def payloads(self) -> tuple[dict[str, JsonValue], ...]:
        return tuple(object_value(json.loads(line)) for body in self.objects().values() for line in body.splitlines())


def s3_config(
    path: Path, sink_url: str, extra: Mapping[str, JsonValue], settings: Mapping[str, JsonValue] | None = None
) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"].update(
        {
            "callbacks": ["s3_v2"],
            "s3_callback_params": {
                "s3_bucket_name": BUCKET,
                "s3_region_name": "us-east-1",
                "s3_endpoint_url": sink_url,
                "s3_path": PREFIX,
                "s3_aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
                "s3_aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                **extra,
            },
            **(settings or {}),
        }
    )
    target: Final = path / "s3_v2.yaml"
    target.write_text(yaml.safe_dump(config))
    return target


def _chat_completion(identity: str) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
    }


def _chat_stream_frames(identity: str) -> tuple[bytes, ...]:
    chunks: Final = (
        {
            "id": identity,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}],
        },
        {
            "id": identity,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
        },
    )
    return tuple(f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks) + (b"data: [DONE]\n\n",)


def _messages_completion(identity: str) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5-20250929",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 11, "output_tokens": 4},
    }


def _messages_stream_frames(identity: str) -> tuple[bytes, ...]:
    events: Final = (
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": identity,
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5-20250929",
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 11, "output_tokens": 1},
                },
            },
        ),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        (
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 4}},
        ),
        ("message_stop", {"type": "message_stop"}),
    )
    return tuple(f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode() for name, payload in events)


def _responses_completion(identity: str) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "id": f"msg_{identity}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
    }


def _responses_stream_frames(identity: str) -> tuple[bytes, ...]:
    events: Final = (
        (
            "response.created",
            {
                "type": "response.created",
                "response": {**_responses_completion(identity), "status": "in_progress", "output": []},
            },
        ),
        (
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": f"msg_{identity}",
                "output_index": 0,
                "content_index": 0,
                "delta": "ok",
            },
        ),
        ("response.completed", {"type": "response.completed", "response": _responses_completion(identity)}),
    )
    return tuple(f"event: {name}\ndata: {json.dumps(payload)}\n\n".encode() for name, payload in events)


def surface_reply(request: Request) -> Reply:
    """Scripted upstream that echoes the caller's marker string back as the response id."""
    if request.method != "POST" or not request.body:
        return Reply(status=404)
    body: Final = json.loads(request.body)
    if request.target.endswith("/chat/completions"):
        identity: Final = body["messages"][0]["content"]
        if body.get("stream"):
            return Reply(content_type="text/event-stream", chunks=_chat_stream_frames(identity))
        return Reply(body=json.dumps(_chat_completion(identity)).encode())
    if request.target.endswith("/messages"):
        identity_messages: Final = body["messages"][0]["content"]
        if body.get("stream"):
            return Reply(content_type="text/event-stream", chunks=_messages_stream_frames(identity_messages))
        return Reply(body=json.dumps(_messages_completion(identity_messages)).encode())
    assert request.target.endswith("/responses"), request.target
    identity_responses: Final = body["input"]
    if body.get("stream"):
        return Reply(content_type="text/event-stream", chunks=_responses_stream_frames(identity_responses))
    return Reply(body=json.dumps(_responses_completion(identity_responses)).encode())


SURFACES: Final = ("chat", "chat_stream", "messages", "messages_stream", "responses", "responses_stream")


def call_surface(
    candidate: Gateway,
    surface: str,
    openai_model: str,
    anthropic_model: str,
    key: str,
    marker: str,
    no_cache: bool = True,
) -> tuple[str, str | None]:
    """Drive one request through the given surface; return (client-visible response id, x-litellm-call-id)."""
    base: Final = str(candidate.client.base_url).rstrip("/")
    headers: Final = {"Authorization": f"Bearer {key}"}
    if surface == "chat":
        reply: Final = openai.OpenAI(base_url=f"{base}/v1", api_key=key).chat.completions.create(
            model=openai_model,
            messages=[{"role": "user", "content": marker}],
            extra_body={"cache": {"no-cache": True}} if no_cache else {},
        )
        return reply.id, None

    async def chat_stream() -> str:
        stream = await openai.AsyncOpenAI(base_url=f"{base}/v1", api_key=key).chat.completions.create(
            model=openai_model,
            messages=[{"role": "user", "content": marker}],
            stream=True,
            extra_body={"cache": {"no-cache": True}} if no_cache else {},
        )
        seen = ""
        async for chunk in stream:
            seen = chunk.id  # rebind-ok: the stream yields one chunk at a time
        return seen

    if surface == "chat_stream":
        return asyncio.run(chat_stream()), None
    if surface in ("messages", "messages_stream"):
        client: Final = anthropic.Anthropic(base_url=base, api_key="anthropic-placeholder", default_headers=headers)
        if surface == "messages":
            reply_messages: Final = client.messages.create(
                model=anthropic_model, max_tokens=16, messages=[{"role": "user", "content": marker}]
            )
            return reply_messages.id, None
        with client.messages.stream(
            model=anthropic_model, max_tokens=16, messages=[{"role": "user", "content": marker}]
        ) as stream:
            final: Final = stream.get_final_message()
            return final.id, None
    if surface == "responses":
        response: Final = candidate.request(
            "POST",
            "/v1/responses",
            {"model": openai_model, "input": marker, **({"cache": {"no-cache": True}} if no_cache else {})},
            key=key,
        )
        assert response.status_code == 200, response.text
        return str(response.json()["id"]), response.headers.get("x-litellm-call-id")
    assert surface == "responses_stream", surface
    with candidate.client.stream(
        "POST",
        "/v1/responses",
        json={"model": openai_model, "input": marker, "stream": True},
        headers=headers,
    ) as response:
        text: Final = response.read().decode()
        assert response.status_code == 200, text
        call_id: Final = response.headers.get("x-litellm-call-id")
    assert marker in text, text
    return marker, call_id


def collect_payloads(sink: RecordingS3Sink, count: int, seconds: float = 60) -> tuple[dict[str, JsonValue], ...]:
    """Wait until `count` stored payload lines exist, then return every stored payload object."""

    def delivered() -> int:
        return sum(len(body.splitlines()) for body in sink.objects().values())

    eventually(delivered, lambda total: total >= count, seconds=seconds)
    return sink.payloads()


def mixed_burst(
    candidate: Gateway, openai_model: str, anthropic_model: str, key: str, marker: str, per_surface: int = 8
) -> tuple[tuple[str, str | None], ...]:
    """Fire `per_surface` requests on every surface; returns (response id, x-litellm-call-id) per request."""
    jobs: Final = tuple(
        (surface, f"{marker}-{surface}-{index}") for surface in SURFACES for index in range(per_surface)
    )

    def call(job: tuple[str, str]) -> tuple[str, str | None]:
        surface, identity = job
        return call_surface(candidate, surface, openai_model, anthropic_model, key, identity)

    with ThreadPoolExecutor(max_workers=48) as pool:
        return tuple(pool.map(call, jobs))


def matched_ids(
    payloads: tuple[dict[str, JsonValue], ...], answered: tuple[tuple[str, str | None], ...]
) -> frozenset[str]:
    """Every payload must be accountable to an answered request by response id or litellm_call_id."""
    response_ids: Final = frozenset(observed for observed, _ in answered)
    call_ids: Final = frozenset(call_id for _, call_id in answered if call_id is not None)
    landed: Final = []
    for payload in payloads:
        if payload["id"] in response_ids:
            landed.append(payload["id"])
            continue
        uncached: Final = str(payload["id"]).rsplit("_cache_hit", 1)[0]
        if uncached in response_ids:
            landed.append(str(payload["id"]))
            continue
        assert payload["litellm_call_id"] in call_ids, f"unmatched payload {payload['id']!r}"
        landed.append(str(payload["id"]))
    return frozenset(landed)
