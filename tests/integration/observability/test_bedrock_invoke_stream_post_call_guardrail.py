"""A post_call guardrail scans a Claude model's Bedrock invoke-with-response-stream passthrough as it relays (LIT-9618).

The caller is boto3 ``InvokeModelWithResponseStream`` (or raw HTTP) against
``/bedrock/model/<model>/invoke-with-response-stream``. Each ``chunk`` frame carries one base64 Anthropic Messages
stream event, and a guardrail that scans responses through ``apply_guardrail`` runs on those events the way it runs
on ``/bedrock/model/<model>/converse-stream`` and ``/v1/messages``: frames reach the caller as the upstream sends
them, every fifth frame and the end of the stream are scanned, a block ends the stream with an AWS exception frame,
and a held stream gets its masks written back into re-encoded ``chunk`` frames. Streams of other model families keep
going out as received"""

from __future__ import annotations

import base64
import json
import os
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import unquote

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

CLAUDE_ID: Final = "anthropic.claude-sonnet-5-v1:0"
LLAMA_ID: Final = "meta.llama4-maverick-17b-instruct-v1:0"
PROFILE_ID: Final = "arn:aws:bedrock:us-east-1:000000000000:application-inference-profile/relayprofile"
CLAUDE: Final = "invoke-relay"
LLAMA: Final = "invoke-relay-llama"
PROFILE: Final = "invoke-relay-profile"
EVENT_STREAM: Final = "application/vnd.amazon.eventstream"
SAMPLED: Final = "invoke-sampled"
HELD: Final = "invoke-held"
END_ONLY: Final = "invoke-end-only"
CODE: Final = "invoke-code"
BEDROCK: Final = "invoke-bedrock"
BEDROCK_MESSAGE: Final = "invoke-bedrock-message"
UNGUARDED: Final = "invoke-unguarded"
CODE_BLOCK_REASON: Final = "code says no"
BLOCK_REASON: Final = "Violated guardrail policy"
BEDROCK_BLOCK_MESSAGE: Final = "Sorry, the model cannot answer this question."
MASKED_WORD: Final = "Paris"
MASK: Final = "[CITY]"
BLOCKED_WORD: Final = "FORBIDDEN"


def chunk_frame(event: Mapping[str, JsonValue]) -> bytes:
    encoded: Final = base64.b64encode(json.dumps(event, separators=(",", ":")).encode()).decode()
    return aws_event_stream_frame(
        {":event-type": "chunk", ":content-type": "application/json", ":message-type": "event"},
        json.dumps({"bytes": encoded, "p": "abcdefghij"}, separators=(",", ":")).encode(),
    )


def claude_frames(*deltas: str) -> tuple[bytes, ...]:
    """A Claude invoke stream with one text content block, one ``text_delta`` event per string."""
    return (
        chunk_frame(
            {
                "type": "message_start",
                "message": {
                    "id": "msg_bdrk_relay",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-5",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 9, "output_tokens": 1},
                },
            }
        ),
        chunk_frame({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        *(
            chunk_frame({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": delta}})
            for delta in deltas
        ),
        chunk_frame({"type": "content_block_stop", "index": 0}),
        chunk_frame(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 4},
            }
        ),
        chunk_frame(
            {
                "type": "message_stop",
                "amazon-bedrock-invocationMetrics": {
                    "inputTokenCount": 9,
                    "outputTokenCount": 4,
                    "invocationLatency": 120,
                    "firstByteLatency": 40,
                },
            }
        ),
    )


def llama_frames(*generations: str) -> tuple[bytes, ...]:
    return tuple(
        chunk_frame({"generation": text, "prompt_token_count": None, "generation_token_count": 1, "stop_reason": None})
        for text in generations
    )


@dataclass(frozen=True, slots=True)
class Frame:
    message_type: str
    kind: str
    payload: Mapping[str, JsonValue]


def _frame(headers: Mapping[str, str], payload: bytes) -> Frame:
    body: Final = json.loads(payload)
    if headers.get(":event-type") != "chunk":
        return Frame(headers[":message-type"], headers.get(":event-type") or headers[":exception-type"], body)
    event: Final = json.loads(base64.b64decode(body["bytes"]))
    return Frame("event", str(event.get("type")), event)


def decode(raw: bytes) -> tuple[Frame, ...]:
    buffer: Final = EventStreamBuffer()
    buffer.add_data(raw)
    return tuple(_frame(message.headers, message.payload) for message in buffer)


def streamed_text(frames: Sequence[Frame]) -> str:
    return "".join(
        str(delta["text"])
        for frame in frames
        if frame.kind == "content_block_delta" and isinstance(delta := frame.payload["delta"], dict)
    )


def verdict(request: Request) -> Reply:
    texts: Final = [str(text) for text in json.loads(request.body)["texts"]]
    if any(BLOCKED_WORD in text for text in texts):
        return Reply(body=json.dumps({"action": "BLOCKED", "blocked_reason": BLOCK_REASON}).encode())
    if any(MASKED_WORD in text for text in texts):
        masked: Final = [text.replace(MASKED_WORD, MASK) for text in texts]
        return Reply(body=json.dumps({"action": "GUARDRAIL_INTERVENED", "texts": masked}).encode())
    return Reply(body=json.dumps({"action": "NONE"}).encode())


def bedrock_texts(request: Request) -> tuple[str, ...]:
    return tuple(str(item["text"]["text"]) for item in json.loads(request.body)["content"])


def bedrock_verdict(request: Request) -> Reply:
    """ApplyGuardrail that anonymizes ``MASKED_WORD`` and blocks ``BLOCKED_WORD``, as Bedrock policies do."""
    texts: Final = bedrock_texts(request)
    if any(BLOCKED_WORD in text for text in texts):
        word: Final = {"match": BLOCKED_WORD, "action": "BLOCKED"}
        return Reply(
            body=json.dumps(
                {
                    "action": "GUARDRAIL_INTERVENED",
                    "outputs": [{"text": BEDROCK_BLOCK_MESSAGE}],
                    "assessments": [{"wordPolicy": {"customWords": [word]}}],
                }
            ).encode()
        )
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


@dataclass(frozen=True, slots=True)
class StreamOutcome:
    events: tuple[str, ...]
    text: str
    error_code: str | None
    error_message: str | None
    rejected_before_stream: str | None


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

    def bedrock_scans(self, guardrail: str, marker: str) -> tuple[tuple[str, ...], ...]:
        return tuple(
            bedrock_texts(request)
            for request in self.sink.drain()
            if request.target.startswith(f"/{guardrail}/") and marker.encode() in request.body
        )

    def invoke_stream(self, guardrail: str, marker: str, model: str = CLAUDE) -> StreamOutcome:
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
            response: Final = client.invoke_model_with_response_stream(
                modelId=model, body=json.dumps(_invoke_body(marker))
            )
        except ClientError as error:
            return StreamOutcome((), "", None, None, str(error))
        events: Final[list[Mapping[str, JsonValue]]] = []
        error: EventStreamError | None = None
        try:
            for event in response["body"]:
                events.append(json.loads(event["chunk"]["bytes"]))
        except EventStreamError as raised:
            error = raised
        frames: Final = tuple(Frame("event", str(event["type"]), event) for event in events)
        return StreamOutcome(
            tuple(frame.kind for frame in frames),
            streamed_text(frames),
            None if error is None else error.response["Error"]["Code"],
            None if error is None else error.response["Error"]["Message"],
            None,
        )


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


def _bedrock_guardrail(name: str, sink: Wire, **settings: JsonValue) -> dict[str, JsonValue]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "bedrock",
            "mode": "post_call",
            "default_on": False,
            "guardrailIdentifier": name,
            "guardrailVersion": "1",
            "aws_access_key_id": "AKIASYNTHETICGUARD",
            "aws_secret_access_key": "synthetic-guardrail-secret",
            "aws_region_name": "us-east-1",
            "aws_bedrock_runtime_endpoint": f"{sink.url}/{name}",
            **settings,
        },
    }


def _deployment(name: str, model_id: str, upstream: Wire) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {
            "model": f"bedrock/{model_id}",
            "api_base": upstream.url,
            "aws_access_key_id": "AKIASYNTHETICRELAY",
            "aws_secret_access_key": "synthetic-relay-secret",
            "aws_region_name": "us-east-1",
        },
    }


def _invoke_body(marker: str) -> dict[str, JsonValue]:
    return {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": [{"type": "text", "text": marker}]}],
    }


def _marker(request: Request) -> str:
    body: Final = json.loads(request.body)
    if "prompt" in body:
        return str(body["prompt"])
    return str(body["messages"][0]["content"][0]["text"])


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    scripts: Final[dict[str, Reply]] = {}

    def provider(request: Request) -> Reply:
        assert unquote(request.target) in (
            f"/model/{CLAUDE_ID}/invoke-with-response-stream",
            f"/model/{CLAUDE_ID}/invoke",
            f"/model/{LLAMA_ID}/invoke-with-response-stream",
            f"/model/{PROFILE_ID}/invoke-with-response-stream",
        ), request.target
        return scripts[_marker(request)]

    def vendor(request: Request) -> Reply:
        if request.target.startswith((f"/{BEDROCK}/", f"/{BEDROCK_MESSAGE}/")):
            return bedrock_verdict(request)
        return verdict(request)

    with httpx.Client(base_url=os.environ["INTEGRATION_PROXY_URL"], timeout=30, trust_env=False) as root_client:
        root: Final = Gateway(
            root_client,
            os.environ.get("INTEGRATION_MASTER_KEY", "sk-integration-master"),
            os.environ["INTEGRATION_UPSTREAM_URL"],
        )
        directory: Final = tmp_path_factory.mktemp("invoke-stream-relay")
        with wire_server(provider) as upstream, wire_server(vendor) as sink:
            config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
            config["guardrails"] = [
                _guardrail(SAMPLED, sink),
                _guardrail(HELD, sink, streaming_buffer_until_moderated=True),
                _guardrail(END_ONLY, sink, streaming_end_of_stream_only=True),
                _bedrock_guardrail(BEDROCK, sink),
                _bedrock_guardrail(BEDROCK_MESSAGE, sink, disable_exception_on_block=True),
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
                _deployment(CLAUDE, CLAUDE_ID, upstream),
                _deployment(LLAMA, LLAMA_ID, upstream),
                _deployment(PROFILE, PROFILE_ID, upstream),
            ]
            config_path: Final = directory / "invoke-stream-relay.yaml"
            config_path.write_text(yaml.safe_dump(config))
            with owned_proxy_process(root, directory, {}, config=config_path, workers=2) as owned:
                with owned.gateway.scenario() as scenario:
                    guarded: Final = (SAMPLED, HELD, END_ONLY, CODE, BEDROCK, BEDROCK_MESSAGE)
                    yield Rig(
                        owned.gateway,
                        scenario,
                        upstream,
                        sink,
                        scripts,
                        {
                            **{name: scenario.key(guardrails=[name]) for name in guarded},
                            UNGUARDED: scenario.key(),
                        },
                    )


def _stream_with_call_id(rig: Rig, guardrail: str, marker: str, model: str = CLAUDE) -> tuple[int, str, bytes, str]:
    with rig.gateway.client.stream(
        "POST",
        f"/bedrock/model/{model}/invoke-with-response-stream",
        json={"prompt": marker} if model == LLAMA else _invoke_body(marker),
        headers={"Authorization": f"Bearer {rig.keys[guardrail]}"},
        timeout=case_timeout_seconds(),
    ) as response:
        return (
            response.status_code,
            response.headers.get("content-type", ""),
            response.read(),
            response.headers.get("x-litellm-call-id", ""),
        )


def _stream(rig: Rig, guardrail: str, marker: str, model: str = CLAUDE) -> tuple[int, str, bytes]:
    status, content_type, body, _ = _stream_with_call_id(rig, guardrail, marker, model)
    return status, content_type, body


def _exception(frames: Sequence[Frame]) -> tuple[str, str] | None:
    last: Final = frames[-1] if frames else None
    if last is None or last.message_type != "exception":
        return None
    return last.kind, str(last.payload.get("message"))


def _spend_statuses(call_id: str, at_least: int) -> list[JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT status FROM "LiteLLM_SpendLogs" WHERE litellm_call_id=%s OR request_id=%s '
            "OR metadata->>'litellm_call_id'=%s ORDER BY status",
            (call_id, call_id, call_id),
        ),
        lambda rows: len(rows) >= at_least,
        seconds=70,
    )
    return [row["status"] for row in rows]


def test_first_frames_reach_the_caller_while_the_upstream_is_still_streaming_and_get_scanned(rig: Rig) -> None:
    gate: Final = threading.Event()
    marker: Final = f"invoke-{uuid.uuid4().hex}"
    frames: Final = claude_frames(f"Hello {marker} ", "world")
    first_transport_chunk: Final = frames[0] + frames[1]
    rig.scripts[marker] = Reply(
        content_type=EVENT_STREAM, chunks=(first_transport_chunk, *frames[2:]), gate_after_first=gate
    )
    release_held_upstream: Final = threading.Timer(import_seconds(), gate.set)
    release_held_upstream.start()
    try:
        with rig.gateway.client.stream(
            "POST",
            f"/bedrock/model/{CLAUDE}/invoke-with-response-stream",
            json=_invoke_body(marker),
            headers={"Authorization": f"Bearer {rig.keys[SAMPLED]}"},
            timeout=case_timeout_seconds(),
        ) as response:
            chunks: Final = response.iter_raw()
            received: Final = bytearray()
            for chunk in chunks:
                received.extend(chunk)
                if len(received) >= len(first_transport_chunk):
                    break
            upstream_still_held: Final = not gate.is_set()
            gate.set()
            received.extend(b"".join(chunks))
            status: Final = response.status_code
    finally:
        release_held_upstream.cancel()
        gate.set()
    assert status == 200, received
    assert upstream_still_held, "the first frames reached the caller only after the upstream finished its stream"
    assert bytes(received) == b"".join(frames), decode(bytes(received))
    assert rig.scans(SAMPLED, marker) == ((f"Hello {marker} world",),)


def test_block_mid_stream_reaches_boto3_as_an_event_stream_exception(rig: Rig) -> None:
    marker: Final = f"invoke-{uuid.uuid4().hex}"
    rig.scripts[marker] = Reply(
        content_type=EVENT_STREAM, chunks=claude_frames(f"The secret {marker} is ", BLOCKED_WORD, ".")
    )
    outcome: Final = rig.invoke_stream(SAMPLED, marker)
    assert outcome.rejected_before_stream is None, outcome
    assert outcome.events[:2] == ("message_start", "content_block_start"), outcome
    assert "message_stop" not in outcome.events, outcome
    assert (outcome.error_code, outcome.error_message) == ("validationException", BLOCK_REASON), outcome


def test_claude_behind_an_application_inference_profile_is_scanned_by_its_request_body(rig: Rig) -> None:
    marker: Final = f"invoke-{uuid.uuid4().hex}"
    rig.scripts[marker] = Reply(
        content_type=EVENT_STREAM, chunks=claude_frames(f"The secret {marker} is ", BLOCKED_WORD, ".")
    )
    outcome: Final = rig.invoke_stream(SAMPLED, marker, model=PROFILE)
    assert outcome.rejected_before_stream is None, outcome
    assert "message_stop" not in outcome.events, outcome
    assert (outcome.error_code, outcome.error_message) == ("validationException", BLOCK_REASON), outcome


def test_clean_stream_reaches_boto3_unchanged_after_its_scan(rig: Rig) -> None:
    marker: Final = f"invoke-{uuid.uuid4().hex}"
    rig.scripts[marker] = Reply(content_type=EVENT_STREAM, chunks=claude_frames(f"Fine {marker} ", "answer"))
    outcome: Final = rig.invoke_stream(SAMPLED, marker)
    assert outcome == StreamOutcome(
        (
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ),
        f"Fine {marker} answer",
        None,
        None,
        None,
    ), outcome
    assert rig.scans(SAMPLED, marker) == ((f"Fine {marker} answer",),)


def test_held_stream_writes_the_guardrail_masks_back_into_the_chunk_frames(rig: Rig) -> None:
    marker: Final = f"invoke-{uuid.uuid4().hex}"
    rig.scripts[marker] = Reply(
        content_type=EVENT_STREAM, chunks=claude_frames(f"{marker} {MASKED_WORD[:3]}", f"{MASKED_WORD[3:]} is big")
    )
    status, content_type, body = _stream(rig, HELD, marker)
    assert (status, content_type) == (200, EVENT_STREAM), body
    frames: Final = decode(body)
    assert [frame.kind for frame in frames] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ], frames
    assert streamed_text(frames) == f"{marker} {MASK} is big", frames
    assert rig.scans(HELD, marker) == ((f"{marker} {MASKED_WORD} is big",),)


def test_held_stream_block_returns_an_error_before_any_frame(rig: Rig) -> None:
    marker: Final = f"invoke-{uuid.uuid4().hex}"
    rig.scripts[marker] = Reply(
        content_type=EVENT_STREAM, chunks=claude_frames(f"The secret {marker} is ", BLOCKED_WORD)
    )
    outcome: Final = rig.invoke_stream(HELD, marker)
    assert outcome.events == () and outcome.rejected_before_stream is not None, outcome
    assert "(400)" in outcome.rejected_before_stream, outcome


def test_bedrock_guardrail_holds_the_stream_and_anonymizes_it(rig: Rig) -> None:
    marker: Final = f"invoke-{uuid.uuid4().hex}"
    rig.scripts[marker] = Reply(
        content_type=EVENT_STREAM, chunks=claude_frames(f"{marker} ", MASKED_WORD, " is", " the", " capital")
    )
    outcome: Final = rig.invoke_stream(BEDROCK, marker)
    assert (outcome.error_code, outcome.rejected_before_stream) == (None, None), outcome
    assert outcome.text == f"{marker} {MASK} is the capital", outcome
    assert rig.bedrock_scans(BEDROCK, marker) == ((f"{marker} {MASKED_WORD} is the capital",),)


def test_bedrock_guardrail_block_message_replaces_a_held_stream(rig: Rig) -> None:
    marker: Final = f"invoke-{uuid.uuid4().hex}"
    rig.scripts[marker] = Reply(
        content_type=EVENT_STREAM, chunks=claude_frames(f"The secret {marker} is ", BLOCKED_WORD)
    )
    outcome: Final = rig.invoke_stream(BEDROCK_MESSAGE, marker)
    assert (outcome.error_code, outcome.rejected_before_stream) == (None, None), outcome
    assert outcome.events[0] == "message_start" and outcome.events[-1] == "message_stop", outcome
    assert outcome.text == BEDROCK_BLOCK_MESSAGE, outcome


def test_end_of_stream_only_relays_every_content_frame_then_blocks(rig: Rig) -> None:
    marker: Final = f"invoke-{uuid.uuid4().hex}"
    frames: Final = claude_frames(f"Late {marker} ", "one", " two", " three", " four", f" {BLOCKED_WORD}")
    rig.scripts[marker] = Reply(content_type=EVENT_STREAM, chunks=frames)
    status, content_type, body = _stream(rig, END_ONLY, marker)
    assert (status, content_type) == (200, EVENT_STREAM), body
    content_frames: Final = b"".join(frames[:-2])
    assert body.startswith(content_frames), decode(body)
    assert decode(body[len(content_frames) :]) == (
        Frame("exception", "validationException", {"message": BLOCK_REASON}),
    ), decode(body)
    assert rig.scans(END_ONLY, marker) == ((f"Late {marker} one two three four {BLOCKED_WORD}",),)


def test_block_raised_mid_stream_logs_the_relayed_spend_and_the_failure(rig: Rig) -> None:
    marker: Final = f"invoke-{uuid.uuid4().hex}"
    rig.scripts[marker] = Reply(
        content_type=EVENT_STREAM,
        chunks=claude_frames(f"The secret {marker} is ", BLOCKED_WORD, *(f" word{index}" for index in range(30))),
        pause_between_chunks=0.05,
    )
    status, content_type, body, call_id = _stream_with_call_id(rig, CODE, marker)
    assert (status, content_type) == (200, EVENT_STREAM), body
    frames: Final = decode(body)
    assert frames[0].kind == "message_start", frames
    assert _exception(frames) == ("validationException", CODE_BLOCK_REASON), frames
    assert call_id, "the relayed stream carries no x-litellm-call-id"
    assert _spend_statuses(call_id, 2) == ["failure", "success"]


def test_non_claude_invoke_stream_is_relayed_as_received_without_a_scan(rig: Rig) -> None:
    marker: Final = f"invoke-{uuid.uuid4().hex}"
    frames: Final = llama_frames(f"Llama {marker} ", BLOCKED_WORD, " end")
    rig.scripts[marker] = Reply(content_type=EVENT_STREAM, chunks=frames)
    status, content_type, body = _stream(rig, SAMPLED, marker, model=LLAMA)
    assert (status, content_type) == (200, EVENT_STREAM), body
    assert body == b"".join(frames), decode(body)
    assert rig.scans(SAMPLED, marker) == ()


def test_invoke_stream_without_a_guardrail_is_relayed_as_received(rig: Rig) -> None:
    marker: Final = f"invoke-{uuid.uuid4().hex}"
    frames: Final = claude_frames(f"Open {marker} ", BLOCKED_WORD, " end")
    rig.scripts[marker] = Reply(content_type=EVENT_STREAM, chunks=frames)
    status, content_type, body = _stream(rig, UNGUARDED, marker)
    assert (status, content_type) == (200, EVENT_STREAM), body
    assert body == b"".join(frames), decode(body)


def test_client_disconnect_stops_the_proxy_reading_the_upstream(rig: Rig) -> None:
    marker: Final = f"invoke-{uuid.uuid4().hex}"
    frames: Final = claude_frames(f"Long {marker} ", *(f" word{index}" for index in range(60)))
    rig.scripts[marker] = Reply(content_type=EVENT_STREAM, chunks=frames, pause_between_chunks=0.1)
    while not rig.provider.disconnected.empty():
        rig.provider.disconnected.get_nowait()
    with rig.gateway.client.stream(
        "POST",
        f"/bedrock/model/{CLAUDE}/invoke-with-response-stream",
        json=_invoke_body(marker),
        headers={"Authorization": f"Bearer {rig.keys[SAMPLED]}"},
        timeout=case_timeout_seconds(),
    ) as response:
        assert response.status_code == 200
        received: Final = bytearray()
        for chunk in response.iter_raw():
            received.extend(chunk)
            if len(received) >= len(b"".join(frames[:3])):
                break
    closed: Final = eventually(
        lambda: rig.provider.disconnected.qsize(), lambda count: count > 0, seconds=4, return_last_on_timeout=True
    )
    assert closed, "the proxy kept reading the upstream after the caller went away"
    assert rig.provider.disconnected.get_nowait() == f"/model/{CLAUDE_ID}/invoke-with-response-stream"


def test_non_streaming_invoke_block_is_unchanged(rig: Rig) -> None:
    marker: Final = f"invoke-{uuid.uuid4().hex}"
    reply: Final = {
        "id": "msg_bdrk_relay",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": [{"type": "text", "text": f"The secret {marker} is {BLOCKED_WORD}"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 9, "output_tokens": 4},
    }
    rig.scripts[marker] = Reply(body=json.dumps(reply).encode())
    response: Final = rig.gateway.client.post(
        f"/bedrock/model/{CLAUDE}/invoke",
        json=_invoke_body(marker),
        headers={"Authorization": f"Bearer {rig.keys[SAMPLED]}"},
        timeout=case_timeout_seconds(),
    )
    assert response.status_code == 400, response.text
    assert BLOCK_REASON in response.text, response.text
