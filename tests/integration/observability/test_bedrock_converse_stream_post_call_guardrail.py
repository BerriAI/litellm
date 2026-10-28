"""A post_call guardrail on the Bedrock Converse streaming passthrough streams frame by frame (LIT-9515).

The caller is boto3 ``ConverseStream`` (or raw HTTP) against ``/bedrock/model/<model>/converse-stream``. A
guardrail that scans responses through ``apply_guardrail`` runs on the stream the way it runs on
``/v1/chat/completions``, ``/v1/messages`` and ``/v1/responses``: frames reach the caller as the upstream sends
them, every fifth frame and the end of the stream are scanned, and a block ends the stream with an AWS
exception frame. ``streaming_buffer_until_moderated: true`` still holds the stream and applies masks per content
block, and a caller leaving before the first held frame stops the proxy reading the upstream. Noma v2 scans carry
``stream`` and, on the last one, the finished ``response``, as chat completions sends them; other guardrails see the
request as before."""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
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

pytestmark: Final = pytest.mark.shares_worker_state

MODEL_ID: Final = "anthropic.claude-sonnet-5-v1:0"
ROUTER_MODEL: Final = "converse-relay"
EVENT_STREAM: Final = "application/vnd.amazon.eventstream"
SAMPLED: Final = "relay-sampled"
BUFFERED: Final = "relay-buffered"
CODE: Final = "relay-code"
FAILING: Final = "relay-failing"
FAILING_HELD: Final = "relay-failing-held"
END_ONLY: Final = "relay-end-only"
OUTAGE: Final = "relay-outage"
STACKED: Final = "relay-stacked"
RELEASE_ON_SCAN: Final = "relay-release-on-scan"
REPLACING: Final = "relay-replacing"
REPLACING_THEN_HELD: Final = "relay-replacing-then-held"
PARALLEL: Final = "relay-parallel"
PARALLEL_FIRST: Final = "relay-parallel-first"
HELD_END_ONLY: Final = "relay-held-end-only"
NOMA: Final = "relay-noma"
ALICE: Final = "relay-alice"
NOMA_THEN_ALICE: Final = "relay-noma-then-alice"
CODE_BLOCK_REASON: Final = "code says no"
BLOCK_REASON: Final = "Violated guardrail policy"
MASKED_WORD: Final = "Paris"
MASK: Final = "[CITY]"
BLOCKED_WORD: Final = "FORBIDDEN"
REPLACED_WORD: Final = "CLASSIFIED"
REPLACEMENT: Final = "That answer was withheld"


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


def parallel_verdict(request: Request) -> Reply:
    """A block-only check that rejects ``MASKED_WORD`` it sees unmasked."""
    texts: Final = [str(text) for text in json.loads(request.body)["texts"]]
    if any(MASKED_WORD in text for text in texts):
        return Reply(body=json.dumps({"action": "BLOCKED", "blocked_reason": BLOCK_REASON}).encode())
    return Reply(body=json.dumps({"action": "NONE"}).encode())


def bedrock_texts(request: Request) -> tuple[str, ...]:
    return tuple(str(item["text"]["text"]) for item in json.loads(request.body)["content"])


def bedrock_verdict(request: Request) -> Reply:
    """ApplyGuardrail anonymizing ``MASKED_WORD``, as a Bedrock sensitive-information policy does."""
    texts: Final = bedrock_texts(request)
    if not any(MASKED_WORD in text for text in texts):
        return Reply(body=json.dumps({"action": "NONE", "outputs": [], "assessments": []}).encode())
    pii: Final = {"match": MASKED_WORD, "type": "ADDRESS", "action": "ANONYMIZED"}
    return Reply(
        body=json.dumps(
            {
                "action": "GUARDRAIL_INTERVENED",
                "outputs": [{"text": text.replace(MASKED_WORD, MASK)} for text in texts],
                "assessments": [{"sensitiveInformationPolicy": {"piiEntities": [pii]}}],
            }
        ).encode()
    )


def bedrock_replacing_verdict(request: Request) -> Reply:
    """ApplyGuardrail blocking ``REPLACED_WORD`` with its blocked message, as a Bedrock word policy does."""
    if not any(REPLACED_WORD in text for text in bedrock_texts(request)):
        return Reply(body=json.dumps({"action": "NONE", "outputs": [], "assessments": []}).encode())
    word: Final = {"match": REPLACED_WORD, "action": "BLOCKED"}
    return Reply(
        body=json.dumps(
            {
                "action": "GUARDRAIL_INTERVENED",
                "outputs": [{"text": REPLACEMENT}],
                "assessments": [{"wordPolicy": {"customWords": [word]}}],
            }
        ).encode()
    )


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    scenario: Scenario
    provider: Wire
    sink: Wire
    scripts: dict[str, Reply]
    keys: Mapping[str, str]
    outage: threading.Event

    def scans(self, guardrail: str, marker: str) -> tuple[tuple[str, ...], ...]:
        return tuple(
            tuple(json.loads(request.body)["texts"])
            for request in self.sink.drain()
            if request.target.startswith(f"/{guardrail}/") and marker.encode() in request.body
        )

    def bedrock_scans(self, guardrail: str, marker: str) -> tuple[tuple[str, ...], ...]:
        return tuple(
            bedrock_texts(request)
            for request in self.sink.drain()
            if request.target.startswith(f"/{guardrail}/") and marker.encode() in request.body
        )

    def forwarded_request_data(self, marker: str) -> dict[str, tuple[Mapping[str, JsonValue], ...]]:
        """The ``request_data`` each response scan forwarded to Noma v2 and Alice, by guardrail, in scan order."""
        forwarded: Final[dict[str, list[Mapping[str, JsonValue]]]] = {NOMA: [], ALICE: []}
        for request in self.sink.drain():
            guardrail = request.target.split("/")[1]
            if guardrail in forwarded and marker.encode() in request.body:
                body = json.loads(request.body)
                if body["input_type"] == "response":
                    forwarded[guardrail].append(body["request_data"])
        return {guardrail: tuple(scans) for guardrail, scans in forwarded.items()}

    def converse_stream(self, guardrail: str, marker: str) -> StreamOutcome:
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
    outage: Final = threading.Event()

    def provider(request: Request) -> Reply:
        assert request.target in (f"/model/{MODEL_ID}/converse-stream", f"/model/{MODEL_ID}/converse"), request.target
        marker: Final = json.loads(request.body)["messages"][0]["content"][0]["text"]
        return scripts[marker]

    def vendor(request: Request) -> Reply:
        failing: Final = request.target.startswith((f"/{FAILING}/", f"/{FAILING_HELD}/"))
        if failing or (outage.is_set() and request.target.startswith(f"/{OUTAGE}/")):
            return Reply(status=503, body=b'{"error":"guardrail vendor unavailable"}')
        if request.target.startswith(f"/{RELEASE_ON_SCAN}/"):
            return bedrock_verdict(request)
        if request.target.startswith(f"/{REPLACING}/"):
            return bedrock_replacing_verdict(request)
        if request.target.startswith(f"/{PARALLEL}/"):
            return parallel_verdict(request)
        if request.target.startswith(f"/{NOMA}/"):
            return Reply(body=json.dumps({"action": "NONE"}).encode())
        if request.target.startswith(f"/{ALICE}/"):
            return Reply(body=json.dumps({"verdict": "ALLOW"}).encode())
        return verdict(request)

    with httpx.Client(base_url=os.environ["INTEGRATION_PROXY_URL"], timeout=30, trust_env=False) as root_client:
        root: Final = Gateway(
            root_client,
            os.environ.get("INTEGRATION_MASTER_KEY", "sk-integration-master"),
            os.environ["INTEGRATION_UPSTREAM_URL"],
        )
        directory: Final = tmp_path_factory.mktemp("converse-stream-relay")
        with wire_server(provider) as upstream, wire_server(vendor) as sink:
            config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
            config["guardrails"] = [
                _guardrail(PARALLEL, sink, run_in_parallel=True),
                _guardrail(SAMPLED, sink),
                {
                    "guardrail_name": REPLACING,
                    "litellm_params": {
                        "guardrail": "bedrock",
                        "mode": "post_call",
                        "default_on": False,
                        "guardrailIdentifier": "relay-replacing",
                        "guardrailVersion": "1",
                        "aws_access_key_id": "AKIASYNTHETICGUARD",
                        "aws_secret_access_key": "synthetic-guardrail-secret",
                        "aws_region_name": "us-east-1",
                        "aws_bedrock_runtime_endpoint": f"{sink.url}/{REPLACING}",
                        "disable_exception_on_block": True,
                        "streaming_buffer_until_moderated": False,
                    },
                },
                _guardrail(BUFFERED, sink, streaming_buffer_until_moderated=True),
                _guardrail(FAILING, sink),
                _guardrail(FAILING_HELD, sink, streaming_buffer_until_moderated=True),
                _guardrail(END_ONLY, sink, streaming_end_of_stream_only=True),
                _guardrail(
                    HELD_END_ONLY, sink, streaming_buffer_until_moderated=True, streaming_end_of_stream_only=True
                ),
                {
                    "guardrail_name": NOMA,
                    "litellm_params": {
                        "guardrail": "noma_v2",
                        "mode": "post_call",
                        "default_on": False,
                        "api_base": f"{sink.url}/{NOMA}",
                        "api_key": "synthetic-noma-key",
                    },
                },
                {
                    "guardrail_name": ALICE,
                    "litellm_params": {
                        "guardrail": "alice",
                        "mode": "post_call",
                        "default_on": False,
                        "api_base": f"{sink.url}/{ALICE}",
                        "api_key": "synthetic-alice-key",
                    },
                },
                _guardrail(OUTAGE, sink),
                {
                    "guardrail_name": RELEASE_ON_SCAN,
                    "litellm_params": {
                        "guardrail": "bedrock",
                        "mode": "post_call",
                        "default_on": False,
                        "guardrailIdentifier": "relay-release-on-scan",
                        "guardrailVersion": "1",
                        "aws_access_key_id": "AKIASYNTHETICGUARD",
                        "aws_secret_access_key": "synthetic-guardrail-secret",
                        "aws_region_name": "us-east-1",
                        "aws_bedrock_runtime_endpoint": f"{sink.url}/{RELEASE_ON_SCAN}",
                        "streaming_buffer_release_on_scan": True,
                    },
                },
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
                    single: Final = (
                        SAMPLED,
                        BUFFERED,
                        CODE,
                        FAILING,
                        FAILING_HELD,
                        END_ONLY,
                        OUTAGE,
                        RELEASE_ON_SCAN,
                        NOMA,
                    )
                    yield Rig(
                        owned.gateway,
                        scenario,
                        upstream,
                        sink,
                        scripts,
                        {
                            **{name: scenario.key(guardrails=[name]) for name in single},
                            STACKED: scenario.key(guardrails=[SAMPLED, BUFFERED]),
                            REPLACING_THEN_HELD: scenario.key(guardrails=[REPLACING, BUFFERED]),
                            PARALLEL_FIRST: scenario.key(guardrails=[PARALLEL, BUFFERED]),
                            NOMA_THEN_ALICE: scenario.key(guardrails=[NOMA, ALICE]),
                        },
                        outage,
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
        'SELECT status, api_base, prompt_tokens FROM "LiteLLM_SpendLogs" WHERE litellm_call_id=%s OR request_id=%s '
        "OR metadata->>'litellm_call_id'=%s ORDER BY status",
        (call_id, call_id, call_id),
    )


def _spend_rows_for_key_alias(alias: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        "SELECT status, prompt_tokens FROM \"LiteLLM_SpendLogs\" WHERE metadata->>'user_api_key_alias'=%s",
        (alias,),
    )


def _stream_releasing_upstream(
    rig: Rig, marker: str, first_chunk_length: int, gate: threading.Event
) -> tuple[int, bool, bytes]:
    """Stream ``marker`` through the sampled guardrail and release the upstream held after its first transport chunk
    once that chunk has reached the caller; report whether the upstream was still held at that point."""
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
            chunks: Final = response.iter_raw()
            received: Final = bytearray()
            for chunk in chunks:
                received.extend(chunk)
                if len(received) >= first_chunk_length:
                    break
            upstream_still_held: Final = not gate.is_set()
            gate.set()
            received.extend(b"".join(chunks))
            return response.status_code, upstream_still_held, bytes(received)
    finally:
        release_held_upstream.cancel()
        gate.set()


def test_first_frames_reach_the_caller_while_the_upstream_is_still_streaming(rig: Rig) -> None:
    gate: Final = threading.Event()
    marker: Final = f"relay-{uuid.uuid4().hex}"
    frames: Final = converse_frames((f"Hello {marker} ", "world"))
    first_transport_chunk: Final = frames[0] + frames[1]
    rig.scripts[marker] = Reply(
        content_type=EVENT_STREAM, chunks=(first_transport_chunk, *frames[2:]), gate_after_first=gate
    )
    status, upstream_still_held, received = _stream_releasing_upstream(rig, marker, len(first_transport_chunk), gate)
    assert status == 200, received
    assert upstream_still_held, "the first frames reached the caller only after the upstream finished its stream"
    assert received == b"".join(frames), decode(received)
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
        content_type=EVENT_STREAM,
        chunks=converse_frames((f"The secret {marker} is ", BLOCKED_WORD, *(f" word{index}" for index in range(30)))),
        pause_between_chunks=0.05,
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
    assert isinstance(rows[1]["prompt_tokens"], int) and rows[1]["prompt_tokens"] > 0, rows


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


def test_a_frame_whose_payload_is_not_utf8_is_relayed_as_sent_and_the_rest_still_scanned(rig: Rig) -> None:
    marker: Final = f"relay-{uuid.uuid4().hex}"
    clean: Final = converse_frames((f"Before {marker} ", "after"))
    garbled: Final = aws_event_stream_frame(
        {":event-type": "contentBlockDelta", ":content-type": "application/json", ":message-type": "event"},
        b'{"contentBlockIndex":0,"delta":{"text":"\xff"}}',
    )
    frames: Final = (*clean[:2], garbled, *clean[2:])
    rig.scripts[marker] = Reply(content_type=EVENT_STREAM, chunks=frames)
    status, content_type, body = _stream(rig, SAMPLED, marker)
    assert (status, content_type) == (200, EVENT_STREAM), body
    assert body == b"".join(frames), body[-300:]
    assert set(rig.scans(SAMPLED, marker)) == {(f"Before {marker} after",)}


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
    assert outcome.rejected_before_stream.endswith(
        f"(ValidationException) when calling the ConverseStream operation: {BLOCK_REASON}"
    ), outcome


def _spend_statuses(call_id: str, at_least: int) -> list[JsonValue]:
    rows: Final = eventually(lambda: _spend_rows(call_id), lambda rows: len(rows) >= at_least, seconds=70)
    return [row["status"] for row in rows]


def _exception(frames: Sequence[Frame]) -> tuple[str, str] | None:
    last: Final = frames[-1] if frames else None
    if last is None or last.message_type != "exception":
        return None
    return last.kind, str(last.payload.get("message"))


def test_upstream_closing_mid_stream_ends_the_relayed_stream_with_an_exception_frame(rig: Rig) -> None:
    marker: Final = f"relay-{uuid.uuid4().hex}"
    gate: Final = threading.Event()
    frames: Final = converse_frames((f"Cut {marker} ", "short", " here"))
    first_transport_chunk: Final = frames[0] + frames[1]
    rig.scripts[marker] = Reply(
        content_type=EVENT_STREAM, chunks=(first_transport_chunk, *frames[2:]), gate_after_first=gate, abort_after=1
    )
    status, upstream_still_held, body = _stream_releasing_upstream(rig, marker, len(first_transport_chunk), gate)
    assert status == 200, body
    assert upstream_still_held, "the first frames reached the caller only after the upstream closed"
    assert body.startswith(first_transport_chunk), decode(body)
    relayed: Final = decode(body)
    exception: Final = _exception(relayed)
    assert exception is not None and exception[0] == "internalServerException", relayed


def test_guardrail_vendor_failing_mid_stream_fails_closed_with_an_exception_frame(rig: Rig) -> None:
    marker: Final = f"relay-{uuid.uuid4().hex}"
    frames: Final = converse_frames((f"Scan {marker} ", "one", " two", " three", " four", " five"))
    rig.scripts[marker] = Reply(content_type=EVENT_STREAM, chunks=frames)
    status, content_type, body = _stream(rig, FAILING, marker)
    assert (status, content_type) == (200, EVENT_STREAM), body
    assert body.startswith(b"".join(frames[:4])), decode(body)
    relayed: Final = decode(body)
    assert "messageStop" not in [frame.kind for frame in relayed], relayed
    exception: Final = _exception(relayed)
    assert exception is not None and exception[0] == "internalServerException", relayed


def test_held_stream_with_the_guardrail_vendor_failing_returns_an_error_before_any_frame(rig: Rig) -> None:
    marker: Final = f"relay-{uuid.uuid4().hex}"
    rig.scripts[marker] = Reply(content_type=EVENT_STREAM, chunks=converse_frames((f"Held {marker} ", "text")))
    status, content_type, body = _stream(rig, FAILING_HELD, marker)
    assert status == 500 and content_type.startswith("application/json"), (status, content_type, body)


def test_end_of_stream_only_relays_every_content_frame_then_blocks(rig: Rig) -> None:
    marker: Final = f"relay-{uuid.uuid4().hex}"
    frames: Final = converse_frames((f"Late {marker} ", "one", " two", " three", " four", f" {BLOCKED_WORD}"))
    rig.scripts[marker] = Reply(content_type=EVENT_STREAM, chunks=frames)
    status, content_type, body = _stream(rig, END_ONLY, marker)
    assert (status, content_type) == (200, EVENT_STREAM), body
    content_frames: Final = b"".join(frames[:-2])
    assert body.startswith(content_frames), decode(body)
    assert decode(body[len(content_frames) :]) == (
        Frame("exception", "validationException", {"message": BLOCK_REASON}),
    ), decode(body)
    assert rig.scans(END_ONLY, marker) == ((f"Late {marker} one two three four {BLOCKED_WORD}",),)


def test_end_of_stream_only_scans_a_clean_stream_once_and_relays_it_unchanged(rig: Rig) -> None:
    marker: Final = f"relay-{uuid.uuid4().hex}"
    frames: Final = converse_frames((f"Fine {marker} ", "one", " two", " three", " four", " five"))
    rig.scripts[marker] = Reply(content_type=EVENT_STREAM, chunks=frames)
    status, _, body = _stream(rig, END_ONLY, marker)
    assert status == 200 and body == b"".join(frames), decode(body)
    assert rig.scans(END_ONLY, marker) == ((f"Fine {marker} one two three four five",),)


def test_held_stream_released_on_scan_stays_buffered_and_applies_masks(rig: Rig) -> None:
    marker: Final = f"relay-{uuid.uuid4().hex}"
    rig.scripts[marker] = Reply(
        content_type=EVENT_STREAM,
        chunks=converse_frames((f"{marker} ", MASKED_WORD, " is", " the", " capital", " of", " France")),
    )
    status, content_type, body = _stream(rig, RELEASE_ON_SCAN, marker)
    assert (status, content_type) == (200, EVENT_STREAM), body
    assert block_texts(decode(body)) == {0: f"{marker} {MASK} is the capital of France"}, decode(body)
    assert rig.bedrock_scans(RELEASE_ON_SCAN, marker) == ((f"{marker} {MASKED_WORD} is the capital of France",),)


def test_sampled_and_held_guardrails_on_one_key_hold_the_stream_and_mask_it(rig: Rig) -> None:
    marker: Final = f"relay-{uuid.uuid4().hex}"
    rig.scripts[marker] = Reply(
        content_type=EVENT_STREAM, chunks=converse_frames((f"{MASKED_WORD} {marker} ", "is", " the", " capital"))
    )
    status, content_type, body = _stream(rig, STACKED, marker)
    assert (status, content_type) == (200, EVENT_STREAM), body
    assert block_texts(decode(body)) == {0: f"{MASK} {marker} is the capital"}, decode(body)
    scanned_by: Final = {
        request.target.split("/")[1] for request in rig.sink.drain() if marker.encode() in request.body
    }
    assert scanned_by == {SAMPLED, BUFFERED}, scanned_by


def test_held_guardrail_still_masks_after_a_sampled_guardrail_replaces_the_rest_of_the_answer(rig: Rig) -> None:
    marker: Final = f"relay-{uuid.uuid4().hex}"
    rig.scripts[marker] = Reply(
        content_type=EVENT_STREAM, chunks=converse_frames((f"{MASKED_WORD} {marker} ", "is", f" {REPLACED_WORD}"))
    )
    status, content_type, body = _stream(rig, REPLACING_THEN_HELD, marker)
    assert (status, content_type) == (200, EVENT_STREAM), body
    frames: Final = decode(body)
    assert block_texts(frames) == {0: f"{MASK} {marker} is {REPLACED_WORD}", 1: REPLACEMENT}, frames
    assert frames[-1] == Frame("event", "messageStop", {"stopReason": "guardrail_intervened"}), frames


def test_parallel_guardrail_listed_before_a_masker_still_scans_the_masked_text(rig: Rig) -> None:
    marker: Final = f"relay-{uuid.uuid4().hex}"
    rig.scripts[marker] = Reply(
        content_type=EVENT_STREAM, chunks=converse_frames((f"{MASKED_WORD} {marker} ", "is the capital"))
    )
    status, content_type, body = _stream(rig, PARALLEL_FIRST, marker)
    assert (status, content_type) == (200, EVENT_STREAM), body
    assert block_texts(decode(body)) == {0: f"{MASK} {marker} is the capital"}, decode(body)
    assert rig.scans(PARALLEL, marker) == ((f"{MASK} {marker} is the capital",),)


def test_client_disconnect_stops_the_proxy_reading_the_upstream(rig: Rig) -> None:
    marker: Final = f"relay-{uuid.uuid4().hex}"
    frames: Final = converse_frames((f"Long {marker} ", *(f" word{index}" for index in range(60))))
    rig.scripts[marker] = Reply(content_type=EVENT_STREAM, chunks=frames, pause_between_chunks=0.1)
    while not rig.provider.disconnected.empty():
        rig.provider.disconnected.get_nowait()
    with rig.gateway.client.stream(
        "POST",
        _stream_path(),
        json=_stream_body(marker),
        headers={"Authorization": f"Bearer {rig.keys[SAMPLED]}"},
        timeout=case_timeout_seconds(),
    ) as response:
        assert response.status_code == 200
        received: Final = bytearray()
        for chunk in response.iter_raw():
            received.extend(chunk)
            if len(received) >= len(b"".join(frames[:3])):
                break
    target: Final = f"/model/{MODEL_ID}/converse-stream"
    closed: Final = eventually(
        lambda: rig.provider.disconnected.qsize(), lambda count: count > 0, seconds=4, return_last_on_timeout=True
    )
    assert closed, "the proxy kept reading the upstream after the caller went away"
    assert rig.provider.disconnected.get_nowait() == target


def test_client_disconnect_logs_prompt_tokens_the_upstream_never_reported(rig: Rig) -> None:
    marker: Final = f"relay-{uuid.uuid4().hex}"
    frames: Final = converse_frames((f"Long {marker} ", *(f" word{index}" for index in range(60))))
    rig.scripts[marker] = Reply(content_type=EVENT_STREAM, chunks=frames, pause_between_chunks=0.1)
    with rig.gateway.client.stream(
        "POST",
        _stream_path(),
        json=_stream_body(marker),
        headers={"Authorization": f"Bearer {rig.keys[SAMPLED]}"},
        timeout=case_timeout_seconds(),
    ) as response:
        assert response.status_code == 200
        call_id: Final = response.headers.get("x-litellm-call-id", "")
        received: Final = bytearray()
        for chunk in response.iter_raw():
            received.extend(chunk)
            if len(received) >= len(b"".join(frames[:3])):
                break
    assert call_id, "the relayed stream carries no x-litellm-call-id"
    rows: Final = eventually(lambda: _spend_rows(call_id), lambda rows: len(rows) >= 1, seconds=70)
    assert [row["status"] for row in rows] == ["success"], rows
    assert isinstance(rows[0]["prompt_tokens"], int) and rows[0]["prompt_tokens"] > 0, rows


def test_client_leaving_a_held_stream_before_its_first_frame_stops_the_proxy_reading_the_upstream(rig: Rig) -> None:
    marker: Final = f"relay-{uuid.uuid4().hex}"
    frames: Final = converse_frames((f"Held {marker} ", *(f" word{index}" for index in range(80))))
    rig.scripts[marker] = Reply(content_type=EVENT_STREAM, chunks=frames, pause_between_chunks=0.1)
    alias: Final = f"held-{marker}"
    key: Final = rig.scenario.key(guardrails=[HELD_END_ONLY], key_alias=alias)
    while not rig.provider.disconnected.empty():
        rig.provider.disconnected.get_nowait()
    with pytest.raises(httpx.ReadTimeout):
        rig.gateway.client.post(
            _stream_path(),
            json=_stream_body(marker),
            headers={"Authorization": f"Bearer {key}"},
            timeout=httpx.Timeout(case_timeout_seconds(), read=1.5),
        )
    closed: Final = eventually(
        lambda: rig.provider.disconnected.qsize(), lambda count: count > 0, seconds=3, return_last_on_timeout=True
    )
    assert closed, "the proxy kept reading the upstream after the caller left a held stream"
    assert rig.provider.disconnected.get_nowait() == f"/model/{MODEL_ID}/converse-stream"
    rows: Final = eventually(lambda: _spend_rows_for_key_alias(alias), lambda rows: len(rows) >= 1, seconds=70)
    assert [row["status"] for row in rows] == ["success"], rows
    assert isinstance(rows[0]["prompt_tokens"], int) and rows[0]["prompt_tokens"] > 0, rows


def _follow_up_status_after_leaving_a_held_stream(rig: Rig, marker: str) -> int:
    """Leave a held stream of ``marker`` before its first frame on a key with one parallel slot, then report the status
    of the next request on that key once the slot is free, or the last status seen when it never frees."""
    key: Final = rig.scenario.key(guardrails=[HELD_END_ONLY], max_parallel_requests=1)
    follow_up: Final = f"relay-{uuid.uuid4().hex}"
    rig.scripts[follow_up] = Reply(content_type=EVENT_STREAM, chunks=converse_frames((f"Next {follow_up} ", " done")))

    def follow_up_status() -> int:
        return rig.gateway.client.post(
            _stream_path(),
            json=_stream_body(follow_up),
            headers={"Authorization": f"Bearer {key}"},
            timeout=case_timeout_seconds(),
        ).status_code

    with pytest.raises(httpx.ReadTimeout):
        rig.gateway.client.post(
            _stream_path(),
            json=_stream_body(marker),
            headers={"Authorization": f"Bearer {key}"},
            timeout=httpx.Timeout(case_timeout_seconds(), read=1.5),
        )
    return eventually(follow_up_status, lambda status: status == 200, seconds=5, return_last_on_timeout=True)


def test_client_leaving_a_held_stream_before_bedrock_sends_a_byte_frees_its_parallel_request_slot(rig: Rig) -> None:
    marker: Final = f"relay-{uuid.uuid4().hex}"
    first_byte: Final = threading.Event()
    rig.scripts[marker] = Reply(
        content_type=EVENT_STREAM, chunks=converse_frames((f"Held {marker} ", " done")), gate_before_first=first_byte
    )
    try:
        status: Final = _follow_up_status_after_leaving_a_held_stream(rig, marker)
    finally:
        first_byte.set()
    assert status == 200, "the request that left before Bedrock answered still holds the key's only parallel slot"


def test_client_leaving_a_held_stream_after_part_of_the_first_frame_frees_its_parallel_request_slot(rig: Rig) -> None:
    marker: Final = f"relay-{uuid.uuid4().hex}"
    frames: Final = converse_frames((f"Held {marker} ", " done"))
    rest_of_stream: Final = threading.Event()
    rig.scripts[marker] = Reply(
        content_type=EVENT_STREAM,
        chunks=(frames[0][:12], frames[0][12:], *frames[1:]),
        gate_after_first=rest_of_stream,
    )
    try:
        status: Final = _follow_up_status_after_leaving_a_held_stream(rig, marker)
    finally:
        rest_of_stream.set()
    assert status == 200, "the request that left with part of Bedrock's first frame still holds the key's only slot"


@pytest.mark.parametrize(
    "word_count", [12, 9], ids=["a-sampled-scan-sees-the-end", "the-last-sampled-scan-runs-before-the-end"]
)
def test_noma_scans_carry_stream_and_only_the_last_carries_the_finished_response(rig: Rig, word_count: int) -> None:
    marker: Final = f"relay-{uuid.uuid4().hex}"
    words: Final = (f"Noma {marker} ", *(f" word{index}" for index in range(word_count - 1)))
    rig.scripts[marker] = Reply(content_type=EVENT_STREAM, chunks=converse_frames(words))
    status, _, body = _stream(rig, NOMA, marker)
    assert status == 200 and _exception(decode(body)) is None, decode(body)
    scans: Final = rig.forwarded_request_data(marker)[NOMA]
    assert len(scans) >= 2, scans
    assert [scan.get("stream") for scan in scans] == [True] * len(scans), scans
    assert [scan.get("response") is not None for scan in scans] == [False] * (len(scans) - 1) + [True], scans
    assert scans[-1]["response"] == {
        "output": {"message": {"role": "assistant", "content": [{"text": "".join(words)}]}},
        "stopReason": "end_turn",
    }, scans[-1]


def test_noma_gets_one_scan_of_a_reply_that_only_calls_a_tool_without_input(rig: Rig) -> None:
    marker: Final = f"relay-{uuid.uuid4().hex}"
    tool_use: Final = {"toolUseId": "tooluse_relay", "name": "get_time", "type": "tool_use"}
    rig.scripts[marker] = Reply(
        content_type=EVENT_STREAM,
        chunks=(
            event_frame("messageStart", {"role": "assistant"}),
            event_frame("contentBlockStart", {"contentBlockIndex": 0, "start": {"toolUse": tool_use}}),
            event_frame("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"toolUse": {"input": ""}}}),
            event_frame("contentBlockStop", {"contentBlockIndex": 0}),
            event_frame("messageStop", {"stopReason": "tool_use"}),
            event_frame("metadata", {"usage": {"inputTokens": 9, "outputTokens": 4, "totalTokens": 13}, "metrics": {}}),
        ),
    )
    status, _, body = _stream(rig, NOMA, marker)
    assert status == 200 and _exception(decode(body)) is None, decode(body)
    finished: Final = {
        "output": {"message": {"role": "assistant", "content": [{"toolUse": {**tool_use, "input": {}}}]}},
        "stopReason": "tool_use",
    }
    scans: Final = rig.forwarded_request_data(marker)[NOMA]
    assert [(scan.get("stream"), scan.get("response")) for scan in scans] == [(True, finished)], scans


def test_guardrails_after_noma_see_the_streamed_request_unchanged(rig: Rig) -> None:
    marker: Final = f"relay-{uuid.uuid4().hex}"
    words: Final = (f"Alice {marker} ", *(f" word{index}" for index in range(11)))
    rig.scripts[marker] = Reply(content_type=EVENT_STREAM, chunks=converse_frames(words))
    status, _, body = _stream(rig, NOMA_THEN_ALICE, marker)
    assert status == 200 and _exception(decode(body)) is None, decode(body)
    scans: Final = rig.forwarded_request_data(marker)[ALICE]
    assert len(scans) >= 2, scans
    assert [(scan.get("stream"), scan.get("response")) for scan in scans] == [(None, None)] * len(scans), scans


def test_non_streaming_converse_block_is_unchanged(rig: Rig) -> None:
    marker: Final = f"relay-{uuid.uuid4().hex}"
    reply: Final = {
        "output": {"message": {"role": "assistant", "content": [{"text": f"The secret {marker} is {BLOCKED_WORD}"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 9, "outputTokens": 4, "totalTokens": 13},
        "metrics": {"latencyMs": 10},
    }
    rig.scripts[marker] = Reply(body=json.dumps(reply).encode())
    response: Final = rig.gateway.client.post(
        f"/bedrock/model/{ROUTER_MODEL}/converse",
        json=_stream_body(marker),
        headers={"Authorization": f"Bearer {rig.keys[SAMPLED]}"},
        timeout=case_timeout_seconds(),
    )
    assert response.status_code == 400, response.text
    assert BLOCK_REASON in response.text, response.text


def test_concurrent_streams_each_get_their_own_verdict_and_spend_rows(rig: Rig) -> None:
    markers: Final = [f"relay-{uuid.uuid4().hex}" for _ in range(16)]
    blocked: Final = frozenset(markers[::2])
    for marker in markers:
        words = (f"Burst {marker} ", "one", " two", BLOCKED_WORD if marker in blocked else " three", " four", " five")
        rig.scripts[marker] = Reply(content_type=EVENT_STREAM, chunks=converse_frames(words), pause_between_chunks=0.02)
    with ThreadPoolExecutor(max_workers=len(markers)) as pool:
        outcomes: Final = dict(zip(markers, pool.map(partial(_stream_with_call_id, rig, SAMPLED), markers)))
    for marker, (status, _, body, call_id) in outcomes.items():
        assert status == 200 and call_id, (marker, status, body)
        frames = decode(body)
        if marker in blocked:
            assert _exception(frames) == ("validationException", BLOCK_REASON), (marker, frames)
        else:
            assert _exception(frames) is None and frames[-1].kind == "metadata", (marker, frames)
    for marker, (_, _, _, call_id) in outcomes.items():
        expected = ["failure", "success"] if marker in blocked else ["success"]
        assert _spend_statuses(call_id, len(expected)) == expected, marker


def test_guardrail_vendor_outage_during_a_burst_fails_closed_then_recovers(rig: Rig) -> None:
    def burst(count: int) -> list[tuple[str, tuple[int, str, bytes, str]]]:
        markers = [f"relay-{uuid.uuid4().hex}" for _ in range(count)]
        for marker in markers:
            rig.scripts[marker] = Reply(
                content_type=EVENT_STREAM,
                chunks=converse_frames((f"Outage {marker} ", "one", " two", " three", " four")),
            )
        with ThreadPoolExecutor(max_workers=count) as pool:
            return list(zip(markers, pool.map(partial(_stream_with_call_id, rig, OUTAGE), markers)))

    rig.outage.set()
    try:
        during: Final = burst(8)
    finally:
        rig.outage.clear()
    after: Final = burst(8)
    for marker, (status, _, body, _) in during:
        exception = _exception(decode(body))
        assert status == 200 and exception is not None and exception[0] == "internalServerException", (marker, body)
    for marker, (status, _, body, _) in after:
        assert status == 200 and _exception(decode(body)) is None, (marker, body)
        assert body == b"".join(rig.scripts[marker].chunks or ()), (marker, decode(body))
    for marker, (_, _, _, call_id) in (*during, *after):
        assert _spend_statuses(call_id, 1).count("success") == 1, marker
