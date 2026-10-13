"""A post_call guardrail on the Bedrock Converse streaming passthrough streams frame by frame (LIT-9515).

The caller is boto3 ``ConverseStream`` (or raw HTTP) against ``/bedrock/model/<model>/converse-stream``. A
guardrail that scans responses through ``apply_guardrail`` runs on the stream the way it runs on
``/v1/chat/completions``, ``/v1/messages`` and ``/v1/responses``: frames reach the caller as the upstream sends
them, every fifth frame and the end of the stream are scanned, and a block ends the stream with an AWS
exception frame. ``streaming_buffer_until_moderated: true`` still holds the stream and applies masks per content
block."""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import boto3
import httpx
import pytest
import yaml
from botocore.config import Config
from botocore.eventstream import EventStreamBuffer
from botocore.exceptions import ClientError, EventStreamError
from integration._support.budget import case_timeout_seconds, import_seconds
from integration._support.client import Gateway, JsonValue, Scenario, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy_process
from integration._support.upstream import aws_event_stream_frame
from integration._support.wire import Reply, Request, Wire, wire_server

MODEL_ID: Final = "anthropic.claude-sonnet-5-v1:0"
ROUTER_MODEL: Final = "converse-relay"
EVENT_STREAM: Final = "application/vnd.amazon.eventstream"
SAMPLED: Final = "relay-sampled"
BUFFERED: Final = "relay-buffered"
CODE: Final = "relay-code"
CODE_BLOCK_REASON: Final = "code says no"
BLOCK_REASON: Final = "Violated guardrail policy"
MASKED_WORD: Final = "Paris"
MASK: Final = "[CITY]"
BLOCKED_WORD: Final = "FORBIDDEN"


def event_frame(event_type: str, payload: Mapping[str, JsonValue]) -> bytes:
    return aws_event_stream_frame(
        {":event-type": event_type, ":content-type": "application/json", ":message-type": "event"},
        json.dumps(payload, separators=(",", ":")).encode(),
    )


def converse_frames(*blocks: Sequence[str]) -> tuple[bytes, ...]:
    """A Converse stream with one text content block per entry of ``blocks``, one delta frame per string."""
    return (
        event_frame("messageStart", {"role": "assistant"}),
        *(
            frame
            for index, deltas in enumerate(blocks)
            for frame in (
                *(event_frame("contentBlockDelta", {"contentBlockIndex": index, "delta": {"text": t}}) for t in deltas),
                event_frame("contentBlockStop", {"contentBlockIndex": index}),
            )
        ),
        event_frame("messageStop", {"stopReason": "end_turn"}),
        event_frame("metadata", {"usage": {"inputTokens": 9, "outputTokens": 4, "totalTokens": 13}, "metrics": {}}),
    )


@dataclass(frozen=True, slots=True)
class Frame:
    message_type: str
    kind: str
    payload: Mapping[str, JsonValue]


def decode(raw: bytes) -> tuple[Frame, ...]:
    buffer: Final = EventStreamBuffer()
    buffer.add_data(raw)
    return tuple(
        Frame(
            str(message.headers[":message-type"]),
            str(message.headers.get(":event-type") or message.headers.get(":exception-type")),
            json.loads(message.payload),
        )
        for message in buffer
    )


def block_texts(frames: Sequence[Frame]) -> dict[int, str]:
    texts: Final[dict[int, str]] = {}
    for frame in frames:
        if frame.kind != "contentBlockDelta":
            continue
        index = frame.payload["contentBlockIndex"]
        delta = frame.payload["delta"]
        assert isinstance(index, int) and isinstance(delta, dict), frame
        texts[index] = texts.get(index, "") + str(delta["text"])
    return texts


def verdict(request: Request) -> Reply:
    texts: Final = [str(text) for text in json.loads(request.body)["texts"]]
    if any(BLOCKED_WORD in text for text in texts):
        return Reply(body=json.dumps({"action": "BLOCKED", "blocked_reason": BLOCK_REASON}).encode())
    if any(MASKED_WORD in text for text in texts):
        masked: Final = [text.replace(MASKED_WORD, MASK) for text in texts]
        return Reply(body=json.dumps({"action": "GUARDRAIL_INTERVENED", "texts": masked}).encode())
    return Reply(body=json.dumps({"action": "NONE"}).encode())


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    scenario: Scenario
    provider: Wire
    sink: Wire
    scripts: dict[str, Reply]
    keys: Mapping[str, str]

    def scans(self, guardrail: str, marker: str) -> tuple[tuple[str, ...], ...]:
        return tuple(
            tuple(json.loads(request.body)["texts"])
            for request in self.sink.drain()
            if request.target.startswith(f"/{guardrail}/") and marker.encode() in request.body
        )

    def converse_stream(self, guardrail: str, marker: str) -> "StreamOutcome":
        client: Final = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            endpoint_url=str(self.gateway.client.base_url).rstrip("/") + "/bedrock",
            aws_access_key_id="AKIASYNTHETICRELAY",
            aws_secret_access_key="synthetic-relay-secret",
            config=Config(read_timeout=case_timeout_seconds(), retries={"max_attempts": 1}),
        )
        key: Final = self.keys[guardrail]

        def authenticate(request: object, **_: object) -> None:
            getattr(request, "headers")["x-litellm-api-key"] = key

        client.meta.events.register("before-send.bedrock-runtime.*", authenticate)
        try:
            response: Final = client.converse_stream(
                modelId=ROUTER_MODEL, messages=[{"role": "user", "content": [{"text": marker}]}]
            )
        except ClientError as error:
            return StreamOutcome((), None, None, str(error))
        events: Final[list[str]] = []
        try:
            for event in response["stream"]:
                events.extend(event)
        except EventStreamError as error:
            return StreamOutcome(
                tuple(events), error.response["Error"]["Code"], error.response["Error"]["Message"], None
            )
        return StreamOutcome(tuple(events), None, None, None)


@dataclass(frozen=True, slots=True)
class StreamOutcome:
    events: tuple[str, ...]
    error_code: str | None
    error_message: str | None
    rejected_before_stream: str | None


def _guardrail(name: str, sink: Wire, **settings: JsonValue) -> dict[str, JsonValue]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "generic_guardrail_api",
            "mode": "post_call",
            "default_on": False,
            "api_base": f"{sink.url}/{name}",
            "api_key": "synthetic-guardrail-key",
            **settings,
        },
    }


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    scripts: Final[dict[str, Reply]] = {}

    def provider(request: Request) -> Reply:
        assert request.target == f"/model/{MODEL_ID}/converse-stream", request.target
        marker: Final = json.loads(request.body)["messages"][0]["content"][0]["text"]
        return scripts[marker]

    with httpx.Client(base_url=os.environ["INTEGRATION_PROXY_URL"], timeout=30, trust_env=False) as root_client:
        root: Final = Gateway(
            root_client,
            os.environ.get("INTEGRATION_MASTER_KEY", "sk-integration-master"),
            os.environ["INTEGRATION_UPSTREAM_URL"],
        )
        directory: Final = tmp_path_factory.mktemp("converse-stream-relay")
        with wire_server(provider) as upstream, wire_server(verdict) as sink:
            config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
            config["guardrails"] = [
                _guardrail(SAMPLED, sink),
                _guardrail(BUFFERED, sink, streaming_buffer_until_moderated=True),
                {
                    "guardrail_name": CODE,
                    "litellm_params": {
                        "guardrail": "custom_code",
                        "mode": "post_call",
                        "default_on": False,
                        "custom_code": (
                            "def apply_guardrail(inputs, request_data, input_type):\n"
                            "    for text in inputs['texts']:\n"
                            f"        if '{BLOCKED_WORD}' in text:\n"
                            f"            return block('{CODE_BLOCK_REASON}')\n"
                            "    return allow()\n"
                        ),
                    },
                },
            ]
            config["model_list"] = [
                {
                    "model_name": ROUTER_MODEL,
                    "litellm_params": {
                        "model": f"bedrock/{MODEL_ID}",
                        "api_base": upstream.url,
                        "aws_access_key_id": "AKIASYNTHETICRELAY",
                        "aws_secret_access_key": "synthetic-relay-secret",
                        "aws_region_name": "us-east-1",
                    },
                }
            ]
            config_path: Final = directory / "converse-stream-relay.yaml"
            config_path.write_text(yaml.safe_dump(config))
            with owned_proxy_process(root, directory, {}, config=config_path, workers=2) as owned:
                with owned.gateway.scenario() as scenario:
                    yield Rig(
                        owned.gateway,
                        scenario,
                        upstream,
                        sink,
                        scripts,
                        {name: scenario.key(guardrails=[name]) for name in (SAMPLED, BUFFERED, CODE)},
                    )


def _stream_path() -> str:
    return f"/bedrock/model/{ROUTER_MODEL}/converse-stream"


def _stream_body(marker: str) -> dict[str, JsonValue]:
    return {"messages": [{"role": "user", "content": [{"text": marker}]}]}


def _stream(rig: Rig, guardrail: str, marker: str) -> tuple[int, str, bytes]:
    status, content_type, body, _ = _stream_with_call_id(rig, guardrail, marker)
    return status, content_type, body


def _stream_with_call_id(rig: Rig, guardrail: str, marker: str) -> tuple[int, str, bytes, str]:
    with rig.gateway.client.stream(
        "POST",
        _stream_path(),
        json=_stream_body(marker),
        headers={"Authorization": f"Bearer {rig.keys[guardrail]}"},
        timeout=case_timeout_seconds(),
    ) as response:
        return (
            response.status_code,
            response.headers.get("content-type", ""),
            response.read(),
            response.headers.get("x-litellm-call-id", ""),
        )


def _spend_rows(call_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT status, api_base FROM "LiteLLM_SpendLogs" WHERE litellm_call_id=%s OR request_id=%s '
        "OR metadata->>'litellm_call_id'=%s ORDER BY status",
        (call_id, call_id, call_id),
    )


def test_first_frames_reach_the_caller_while_the_upstream_is_still_streaming(rig: Rig) -> None:
    gate: Final = threading.Event()
    marker: Final = f"relay-{uuid.uuid4().hex}"
    frames: Final = converse_frames((f"Hello {marker} ", "world"))
    first_transport_chunk: Final = frames[0] + frames[1]
    rig.scripts[marker] = Reply(
        content_type=EVENT_STREAM, chunks=(first_transport_chunk, *frames[2:]), gate_after_first=gate
    )
    release_held_upstream: Final = threading.Timer(import_seconds(), gate.set)
    release_held_upstream.start()
    try:
        with rig.gateway.client.stream(
            "POST",
            _stream_path(),
            json=_stream_body(marker),
            headers={"Authorization": f"Bearer {rig.keys[SAMPLED]}"},
            timeout=case_timeout_seconds(),
        ) as response:
            status: Final = response.status_code
            chunks: Final = response.iter_raw()
            received: Final = bytearray()
            for chunk in chunks:
                received.extend(chunk)
                if len(received) >= len(first_transport_chunk):
                    break
            upstream_still_held: Final = not gate.is_set()
            gate.set()
            received.extend(b"".join(chunks))
    finally:
        release_held_upstream.cancel()
        gate.set()
    assert status == 200, bytes(received)
    assert upstream_still_held, "the first frames reached the caller only after the upstream finished its stream"
    assert bytes(received) == b"".join(frames), decode(bytes(received))
    assert rig.scans(SAMPLED, marker) == ((f"Hello {marker} world",),)


def test_block_mid_stream_reaches_boto3_as_an_event_stream_exception(rig: Rig) -> None:
    marker: Final = f"relay-{uuid.uuid4().hex}"
    rig.scripts[marker] = Reply(
        content_type=EVENT_STREAM, chunks=converse_frames((f"The secret {marker} is ", BLOCKED_WORD, "."))
    )
    outcome: Final = rig.converse_stream(SAMPLED, marker)
    assert outcome.rejected_before_stream is None, outcome
    assert outcome.events[:2] == ("messageStart", "contentBlockDelta"), outcome
    assert "messageStop" not in outcome.events, outcome
    assert (outcome.error_code, outcome.error_message) == ("validationException", BLOCK_REASON), outcome


def test_block_raised_mid_stream_logs_the_relayed_spend_and_the_failure(rig: Rig) -> None:
    marker: Final = f"relay-{uuid.uuid4().hex}"
    rig.scripts[marker] = Reply(
        content_type=EVENT_STREAM, chunks=converse_frames((f"The secret {marker} is ", BLOCKED_WORD, "."))
    )
    status, content_type, body, call_id = _stream_with_call_id(rig, CODE, marker)
    assert (status, content_type) == (200, EVENT_STREAM), body
    frames: Final = decode(body)
    assert frames[0].kind == "messageStart", frames
    assert frames[-1].message_type == "exception" and frames[-1].kind == "validationException", frames
    assert CODE_BLOCK_REASON in str(frames[-1].payload["message"]), frames
    assert call_id, "the relayed stream carries no x-litellm-call-id"
    rows: Final = eventually(lambda: _spend_rows(call_id), lambda rows: len(rows) >= 2, seconds=70)
    assert [row["status"] for row in rows] == ["failure", "success"], rows
    assert rows[1]["api_base"], rows


def test_clean_stream_reaches_boto3_unchanged(rig: Rig) -> None:
    marker: Final = f"relay-{uuid.uuid4().hex}"
    rig.scripts[marker] = Reply(content_type=EVENT_STREAM, chunks=converse_frames((f"Fine {marker} ", "answer")))
    outcome: Final = rig.converse_stream(SAMPLED, marker)
    assert outcome == StreamOutcome(
        (
            "messageStart",
            "contentBlockDelta",
            "contentBlockDelta",
            "contentBlockStop",
            "messageStop",
            "metadata",
        ),
        None,
        None,
        None,
    ), outcome
    assert rig.scans(SAMPLED, marker) == ((f"Fine {marker} answer",),)


def test_sampled_stream_forwards_frames_as_sent_without_applying_masks_like_chat(rig: Rig) -> None:
    marker: Final = f"relay-{uuid.uuid4().hex}"
    frames: Final = converse_frames((f"{MASKED_WORD} {marker} ", "is the capital"))
    rig.scripts[marker] = Reply(content_type=EVENT_STREAM, chunks=frames)
    status, content_type, body = _stream(rig, SAMPLED, marker)
    assert (status, content_type) == (200, EVENT_STREAM), body
    assert body == b"".join(frames), decode(body)
    assert rig.scans(SAMPLED, marker) == ((f"{MASKED_WORD} {marker} is the capital",),)


def test_buffered_stream_applies_masks_per_content_block(rig: Rig) -> None:
    marker: Final = f"relay-{uuid.uuid4().hex}"
    rig.scripts[marker] = Reply(
        content_type=EVENT_STREAM,
        chunks=converse_frames((f"{marker} {MASKED_WORD[:3]}", f"{MASKED_WORD[3:]} is big"), ("I like ", MASKED_WORD)),
    )
    status, content_type, body = _stream(rig, BUFFERED, marker)
    assert (status, content_type) == (200, EVENT_STREAM), body
    frames: Final = decode(body)
    assert [frame.kind for frame in frames] == [
        "messageStart",
        "contentBlockDelta",
        "contentBlockDelta",
        "contentBlockStop",
        "contentBlockDelta",
        "contentBlockDelta",
        "contentBlockStop",
        "messageStop",
        "metadata",
    ], frames
    assert block_texts(frames) == {0: f"{marker} {MASK} is big", 1: f"I like {MASK}"}, frames
    assert rig.scans(BUFFERED, marker) == ((f"{marker} {MASKED_WORD} is big", f"I like {MASKED_WORD}"),)


def test_buffered_stream_ending_in_an_upstream_exception_still_applies_masks(rig: Rig) -> None:
    marker: Final = f"relay-{uuid.uuid4().hex}"
    upstream_error: Final = aws_event_stream_frame(
        {
            ":exception-type": "modelStreamErrorException",
            ":content-type": "application/json",
            ":message-type": "exception",
        },
        json.dumps({"message": "Model stream error"}).encode(),
    )
    rig.scripts[marker] = Reply(
        content_type=EVENT_STREAM, chunks=(*converse_frames((f"{marker} ", MASKED_WORD))[:3], upstream_error)
    )
    status, content_type, body = _stream(rig, BUFFERED, marker)
    assert (status, content_type) == (200, EVENT_STREAM), body
    frames: Final = decode(body)
    assert block_texts(frames) == {0: f"{marker} {MASK}"}, frames
    assert frames[-1] == Frame("exception", "modelStreamErrorException", {"message": "Model stream error"}), frames


def test_buffered_stream_block_returns_an_error_before_any_frame(rig: Rig) -> None:
    marker: Final = f"relay-{uuid.uuid4().hex}"
    rig.scripts[marker] = Reply(
        content_type=EVENT_STREAM, chunks=converse_frames((f"The secret {marker} is ", BLOCKED_WORD))
    )
    outcome: Final = rig.converse_stream(BUFFERED, marker)
    assert outcome.events == () and outcome.rejected_before_stream is not None, outcome
    assert "(400)" in outcome.rejected_before_stream, outcome
