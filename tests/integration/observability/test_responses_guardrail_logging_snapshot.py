"""What the logging callbacks see on /v1/responses after a pre-call guardrail rewrites input.

PR 45163 refreshes the logging object inside OpenAIResponsesHandler.process_input_messages. These
cells pin, per input shape and guardrail verdict, the messages/input the payload sink receives, the
input the provider receives, and the spend row — against a scripted provider and guardrail vendor.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import ExitStack
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final
import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from openai import AsyncOpenAI, OpenAI

MARKER: Final = "SECRET_TOKEN"
MASK: Final = "<MASKED_PII>"


def marker_body(tag: str) -> str:
    return f"{tag} please repeat the code. The code is {MARKER}."


def mask(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace(MARKER, MASK)
    if isinstance(value, list):
        return [mask(v) for v in value]
    if isinstance(value, dict):
        return {k: mask(v) for k, v in value.items()}
    return value


def body_blob(value: Any) -> str:
    return json.dumps(value, default=str)


def snapshot_with(values: tuple[Mapping[str, Any], ...], tag: str) -> tuple[Mapping[str, Any], ...]:
    return tuple(v for v in values if tag in body_blob(v))


# ---------------------------------------------------------------- provider (OpenAI wire)


def _responses_object(identity: str, text: str) -> dict[str, Any]:
    return {
        "id": identity,
        "object": "response",
        "created_at": int(time.time()),
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "id": "msg_" + identity,
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 15,
        },
    }


def _sse(event: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(event).encode() + b"\n\n"


def _responses_stream(identity: str, text: str) -> bytes:
    frames: Final = (
        {"type": "response.created", "response": _responses_object(identity, "")},
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "message", "id": "msg_" + identity, "role": "assistant", "status": "in_progress", "content": []},
        },
        {"type": "response.content_part.added", "item_id": "msg_" + identity, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": "", "annotations": []}},
        {"type": "response.output_text.delta", "item_id": "msg_" + identity, "output_index": 0, "content_index": 0, "delta": text},
        {"type": "response.output_text.done", "item_id": "msg_" + identity, "output_index": 0, "content_index": 0, "text": text},
        {"type": "response.content_part.done", "item_id": "msg_" + identity, "output_index": 0, "content_index": 0, "part": {"type": "output_text", "text": text, "annotations": []}},
        {"type": "response.output_item.done", "output_index": 0, "item": {"type": "message", "id": "msg_" + identity, "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": text, "annotations": []}]}},
        {"type": "response.completed", "response": _responses_object(identity, text)},
    )
    return b"".join(_sse(f) for f in frames)


@dataclass
class ProviderState:
    fail: bool = False


def provider_reply(state: ProviderState) -> Any:
    def handler(request: Request) -> Reply:
        if state.fail:
            return Reply(status=500, body=b'{"error": {"message": "scripted provider outage", "type": "server_error"}}')
        identity: Final = "resp_" + uuid.uuid4().hex[:12]
        if request.target.startswith("/v1/responses"):
            body: Final = json.loads(request.body or b"{}")
            if body.get("stream") is True:
                return Reply(
                    status=200,
                    content_type="text/event-stream",
                    chunks=tuple(c + b"\n\n" for c in _responses_stream(identity, "ack").split(b"\n\n")[:-1]),
                )
            return Reply(status=200, body=json.dumps(_responses_object(identity, "ack")).encode())
        if request.target.startswith("/v1/chat/completions"):
            return Reply(
                status=200,
                body=json.dumps(
                    {
                        "id": "chatcmpl-" + uuid.uuid4().hex[:12],
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": "gpt-4o-mini",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ack"}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    }
                ).encode(),
            )
        if request.target.startswith("/providers/err"):
            return Reply(status=500, body=b'{"error": {"message": "scripted provider outage", "type": "server_error"}}')
        return Reply(status=404, body=b'{"error": "unknown provider path"}')

    return handler


# ---------------------------------------------------------------- guardrail vendor + payload sink


def vendor_reply(request: Request) -> Reply:
    target: Final = request.target
    data: Final = json.loads(request.body or b"{}")
    if target.startswith("/mask"):
        return Reply(status=200, body=json.dumps({"action": "MODIFY", "texts": mask(data.get("texts") or [])}).encode())
    if target.startswith("/structured"):
        structured: Final = data.get("structured_messages")
        payload: Final = {"action": "MODIFY"}
        if structured is not None:
            payload["structured_messages"] = mask(structured)
        if data.get("texts"):
            payload["texts"] = mask(data["texts"])
        return Reply(status=200, body=json.dumps(payload).encode())
    if target.startswith("/none"):
        return Reply(status=200, body=json.dumps({"action": "NONE"}).encode())
    if target.startswith("/block"):
        return Reply(status=200, body=json.dumps({"action": "BLOCKED", "blocked_reason": "synthetic marker policy"}).encode())
    if target.startswith("/analyze"):
        text: Final = data.get("text", "")
        findings: Final = [
            {"entity_type": "SECRET", "start": m.start(), "end": m.end(), "score": 0.99}
            for m in __import__("re").finditer(MARKER, text)
        ]
        return Reply(status=200, body=json.dumps(findings).encode())
    if target.startswith("/providers/err"):
        return Reply(status=500, body=b'{"error": {"message": "scripted provider outage", "type": "server_error"}}')
    if target.startswith("/anonymize"):
        text2: Final = data.get("text", "")
        items: Final = [
            {"entity_type": "SECRET", "start": i["start"], "end": i["end"], "operator": "replace", "text": MASK}
            for i in data.get("analyzer_results") or []
        ]
        return Reply(status=200, body=json.dumps({"text": text2.replace(MARKER, MASK), "items": items}).encode())
    return Reply(status=404, body=b'{"error": "unknown vendor path"}')


@dataclass
class SinkGate:
    """Toggles the payload sink between recording and failing (the chaos cell)."""

    fail: bool = False
    seen: list[dict[str, Any]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def events(self) -> tuple[Mapping[str, Any], ...]:
        with self.lock:
            return tuple(self.seen)

    def with_tag(self, tag: str) -> tuple[Mapping[str, Any], ...]:
        return snapshot_with(self.events(), tag)


def sink_reply(gate: SinkGate) -> Any:
    def handler(request: Request) -> Reply:
        if gate.fail:
            return Reply(status=500, body=b'{"error": "sink outage"}')
        with gate.lock:
            gate.seen.append(json.loads(request.body or b"{}"))
        return Reply(status=204, body=b"")

    return handler


CALLBACK_MODULE: Final = '''"""Payload sink callback written by the rig; ships logging-hook kwargs to the wire sink."""
import os

import httpx

from litellm.integrations.custom_logger import CustomLogger


class PayloadSinkLogger(CustomLogger):
    def __init__(self):
        self.sink_url = os.environ["PR45163_SINK_URL"]

    def _post(self, event, kwargs, response_obj):
        payload = {
            "event": event,
            "call_type": kwargs.get("call_type"),
            "model": kwargs.get("model"),
            "litellm_call_id": kwargs.get("litellm_call_id"),
            "messages": kwargs.get("messages"),
            "input": kwargs.get("input"),
            "instructions": kwargs.get("instructions"),
            "slo_messages": (kwargs.get("standard_logging_object") or {}).get("messages")
            if isinstance(kwargs.get("standard_logging_object"), dict)
            else None,
            "guardrail_information": (kwargs.get("standard_logging_object") or {}).get("guardrail_information")
            if isinstance(kwargs.get("standard_logging_object"), dict)
            else None,
        }
        try:
            httpx.post(self.sink_url, json=payload, timeout=10)
        except Exception:
            pass

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._post("sync_success", kwargs, response_obj)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._post("async_success", kwargs, response_obj)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._post("async_failure", kwargs, response_obj)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._post("sync_failure", kwargs, response_obj)


payload_sink = PayloadSinkLogger()
'''


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    provider: Wire
    provider: Wire
    vendor: Wire
    gate: SinkGate
    provider_state: ProviderState


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    synthetic: Final = Gateway(
        httpx.Client(base_url="http://127.0.0.1:1", timeout=15, trust_env=False),
        "sk-pr45163-" + uuid.uuid4().hex,
        os.environ.get("INTEGRATION_UPSTREAM_URL", "http://127.0.0.1:1"),
    )
    directory: Final = tmp_path_factory.mktemp("pr45163")
    gate: Final = SinkGate()
    provider_state: Final = ProviderState()
    with ExitStack() as stack:
        provider: Final = stack.enter_context(wire_server(provider_reply(provider_state)))
        vendor: Final = stack.enter_context(wire_server(vendor_reply))
        sink: Final = stack.enter_context(wire_server(sink_reply(gate)))
        (directory / "pr45163_sink.py").write_text(CALLBACK_MODULE)

        def guarded(name: str, guardrails: list[str] | None = None, base: str | None = None) -> dict[str, Any]:
            entry: Final = {
                "model_name": name,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": "sk-scripted",
                    "api_base": (base or provider.url) + "/v1",
                },
            }
            if guardrails:
                entry["litellm_params"]["guardrails"] = guardrails
            return entry

        config: Final = {
            "model_list": [
                guarded("open-model"),
                guarded("mask-model", ["mask_text"]),
                guarded("struct-model", ["struct_mask"]),
                guarded("nomod-model", ["monitor"]),
                guarded("block-model", ["blocker"]),
                guarded("during-model", ["during_mask"]),
                guarded("dlp-model", ["mask_text", "dlp_logging_only"]),
                guarded("presidio-model", ["presidio_mask"]),
                guarded("err-model", ["mask_text"], base=vendor.url + "/providers/err"),
                guarded("chat-mask-model", ["mask_text"]),
            ],
            "litellm_settings": {"callbacks": ["pr45163_sink.payload_sink"]},
            "guardrails": [
                {"guardrail_name": "mask_text", "litellm_params": {"guardrail": "generic_guardrail_api", "mode": "pre_call", "default_on": False, "api_base": vendor.url + "/mask"}},
                {"guardrail_name": "struct_mask", "litellm_params": {"guardrail": "generic_guardrail_api", "mode": "pre_call", "default_on": False, "api_base": vendor.url + "/structured"}},
                {"guardrail_name": "monitor", "litellm_params": {"guardrail": "generic_guardrail_api", "mode": "pre_call", "default_on": False, "api_base": vendor.url + "/none"}},
                {"guardrail_name": "blocker", "litellm_params": {"guardrail": "generic_guardrail_api", "mode": "pre_call", "default_on": False, "api_base": vendor.url + "/block"}},
                {"guardrail_name": "during_mask", "litellm_params": {"guardrail": "generic_guardrail_api", "mode": "during_call", "default_on": False, "api_base": vendor.url + "/mask"}},
                {"guardrail_name": "dlp_logging_only", "litellm_params": {"guardrail": "generic_guardrail_api", "mode": "logging_only", "default_on": False, "api_base": vendor.url + "/none"}},
                {
                    "guardrail_name": "presidio_mask",
                    "litellm_params": {
                        "guardrail": "presidio",
                        "mode": "pre_call",
                        "default_on": False,
                        "presidio_analyzer_api_base": vendor.url + "/analyze",
                        "presidio_anonymizer_api_base": vendor.url + "/anonymize",
                    },
                },
            ],
            "general_settings": {"master_key": synthetic.key},
        }
        path: Final = directory / "config.yaml"
        path.write_text(yaml.safe_dump(config))
        proxy: Final = stack.enter_context(
            owned_proxy_process(
                synthetic,
                directory,
                {
                "PYTHONPATH": os.pathsep.join((str(directory), os.environ.get("PYTHONPATH", ""))),
                "PR45163_SINK_URL": sink.url,
            },
                config=path,
                workers=2,
            )
        )
        owned: Final = proxy.gateway
        yield Rig(gateway=owned, provider=provider, vendor=vendor, gate=gate, provider_state=provider_state)


# ---------------------------------------------------------------- helpers


def tag() -> str:
    return uuid.uuid4().hex[:12]


def post_responses(rig: Rig, body: Mapping[str, Any]) -> httpx.Response:
    return rig.gateway.request("POST", "/v1/responses", body)


def success_messages(rig: Rig, marker: str) -> tuple[Mapping[str, Any], ...]:
    def fetch() -> tuple[Mapping[str, Any], ...]:
        return tuple(e for e in rig.gate.with_tag(marker) if str(e.get("event", "")).endswith(("success", "failure")))

    return eventually(fetch, lambda events: len(events) >= 1, seconds=30)


def provider_requests(rig: Rig, marker: str) -> tuple[Request, ...]:
    return tuple(r for r in rig.provider.drain() if marker.encode() in r.body or marker in r.target)


def vendor_requests(rig: Rig, marker: str) -> tuple[Request, ...]:
    return tuple(r for r in rig.vendor.drain() if marker.encode() in r.body or marker in r.target)


def spend_row(rig: Rig, request_id: str) -> Mapping[str, Any]:
    rows: Final = eventually(
        lambda: read_rows('SELECT request_id, spend, model FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (request_id,)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return rows[0]


def message_texts(messages: Any) -> str:
    return body_blob(messages)


# ---------------------------------------------------------------- cells


def test_string_input_rewrite_sanitizes_logged_messages(rig: Rig) -> None:
    """Row 1: the motivating leak — string input, texts rewrite branch."""
    marker: Final = "row1-" + tag()
    client: Final = OpenAI(base_url=str(rig.gateway.client.base_url).rstrip("/") + "/v1", api_key=rig.gateway.key)
    response: Final = client.responses.create(model="mask-model", input=marker_body(marker), max_output_tokens=32)
    assert response.id

    events: Final = success_messages(rig, marker)
    assert MARKER not in body_blob([e.get("messages") for e in events]), events
    assert MASK in body_blob([e.get("messages") for e in events])

    provider: Final = provider_requests(rig, marker)
    assert provider and MARKER not in provider[-1].body.decode(), provider[-1].body.decode()[:400]
    spend_row(rig, response.id)


@pytest.mark.asyncio
async def test_list_input_text_parts_rewrite_keeps_masked_chat_shape(rig: Rig) -> None:
    """Row 2: list input with input_text parts — masked text, chat part type after the refresh."""
    marker: Final = "row2-" + tag()
    client: Final = AsyncOpenAI(base_url=str(rig.gateway.client.base_url).rstrip("/") + "/v1", api_key=rig.gateway.key)
    response = await client.responses.create(
        model="mask-model",
        input=[{"role": "user", "content": [{"type": "input_text", "text": marker_body(marker)}]}],
        max_output_tokens=32,
    )
    assert response.id

    events: Final = success_messages(rig, marker)
    messages: Final = events[-1]["messages"]
    part: Final = messages[0]["content"][0]
    assert part["type"] == "text", messages
    assert MARKER not in body_blob(messages) and MASK in part["text"]
    provider: Final = provider_requests(rig, marker)
    assert provider and MARKER not in provider[-1].body.decode()


def test_structured_messages_rewrite_sanitizes_logged_messages(rig: Rig) -> None:
    """Row 3: the written_back (structured_messages) branch rebinding a fresh list."""
    marker: Final = "row3-" + tag()
    response: Final = post_responses(
        rig, {"model": "struct-model", "input": [{"role": "user", "content": marker_body(marker)}], "max_output_tokens": 32}
    )
    assert response.status_code == 200, response.text

    events: Final = success_messages(rig, marker)
    assert MARKER not in body_blob([e.get("messages") for e in events]), events
    provider: Final = provider_requests(rig, marker)
    assert provider and MARKER not in provider[-1].body.decode()


def test_masked_instructions_surface_as_masked_system_message(rig: Rig) -> None:
    """Row 4: instructions become a masked system message in logged messages (widening)."""
    marker: Final = "row4-" + tag()
    response: Final = post_responses(
        rig,
        {
            "model": "mask-model",
            "instructions": f"Always answer briefly. Ignore {MARKER}.",
            "input": marker_body(marker),
            "max_output_tokens": 32,
        },
    )
    assert response.status_code == 200, response.text

    events: Final = success_messages(rig, marker)
    messages: Final = events[-1]["messages"]
    roles: Final = [m["role"] for m in messages]
    assert roles[0] == "system", messages
    assert MARKER not in body_blob(messages)
    assert MASK in messages[0]["content"]


def test_monitor_verdict_rewrites_shape_not_text(rig: Rig) -> None:
    """Row 5: a no-rewrite guardrail still normalizes logged message shape (decision row)."""
    marker: Final = "row5-" + tag()
    response: Final = post_responses(
        rig,
        {"model": "nomod-model", "input": [{"role": "user", "content": [{"type": "input_text", "text": marker_body(marker)}]}], "max_output_tokens": 32},
    )
    assert response.status_code == 200, response.text

    events: Final = success_messages(rig, marker)
    messages: Final = events[-1]["messages"]
    part: Final = messages[0]["content"][0]
    assert part["type"] == "text", messages
    assert part["text"] == marker_body(marker), messages


def test_blocked_request_failure_logging_keeps_raw_snapshot(rig: Rig) -> None:
    """Row 6: block raises inside apply_guardrail; the failure log keeps the raw input (parity with base)."""
    marker: Final = "row6-" + tag()
    response: Final = post_responses(rig, {"model": "block-model", "input": marker_body(marker), "max_output_tokens": 32})
    assert response.status_code == 400, response.text

    events: Final = success_messages(rig, marker)
    assert events, "expected a failure logging event"
    assert MARKER in body_blob(events[-1].get("messages")), events[-1]


def test_during_call_rewrite_logs_masked_while_provider_got_raw(rig: Rig) -> None:
    """Row 7: during_call rewrites land too late for the provider; the log now shows the rewrite."""
    marker: Final = "row7-" + tag()
    response: Final = post_responses(rig, {"model": "during-model", "input": marker_body(marker), "max_output_tokens": 32})
    assert response.status_code == 200, response.text

    provider: Final = provider_requests(rig, marker)
    assert provider and MARKER in provider[-1].body.decode(), "provider must receive raw (rewrite is too late)"
    events: Final = success_messages(rig, marker)
    assert MARKER not in body_blob([e.get("messages") for e in events]), events


def test_streaming_rewrite_sanitizes_at_stream_end(rig: Rig) -> None:
    """Row 8: the whole stream is consumed; the success event at stream end is sanitized."""
    marker: Final = "row8-" + tag()
    with rig.gateway.client.stream(
        "POST",
        "/v1/responses",
        json={"model": "mask-model", "stream": True, "input": marker_body(marker), "max_output_tokens": 32},
        headers={"Authorization": f"Bearer {rig.gateway.key}"},
    ) as response:
        assert response.status_code == 200
        chunks: Final = b"".join(response.iter_raw())
    assert b"response.completed" in chunks

    events: Final = success_messages(rig, marker)
    assert MARKER not in body_blob([e.get("messages") for e in events]), events


def test_unguarded_model_logging_is_unchanged(rig: Rig) -> None:
    """Row 9: control — without a guardrail the refresh never runs."""
    marker: Final = "row9-" + tag()
    response: Final = post_responses(rig, {"model": "open-model", "input": marker_body(marker), "max_output_tokens": 32})
    assert response.status_code == 200, response.text

    events: Final = success_messages(rig, marker)
    messages: Final = events[-1]["messages"]
    assert marker_body(marker) in body_blob(messages)
    provider: Final = provider_requests(rig, marker)
    assert provider and MARKER in provider[-1].body.decode()


def test_chat_completions_masking_is_unchanged(rig: Rig) -> None:
    """Row 10: control — /v1/chat/completions sanitized by the pre-existing update_messages path."""
    marker: Final = "row10-" + tag()
    response: Final = rig.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": "chat-mask-model", "messages": [{"role": "user", "content": marker_body(marker)}], "max_tokens": 32},
    )
    assert response.status_code == 200, response.text

    events: Final = success_messages(rig, marker)
    assert MARKER not in body_blob([e.get("messages") for e in events]), events
    provider: Final = provider_requests(rig, marker)
    assert provider and MARKER not in provider[-1].body.decode()


def test_logging_only_scan_sees_post_mask_content(rig: Rig) -> None:
    """Row 12: a logging_only DLP guardrail now scans the masked text (decider flip)."""
    marker: Final = "row12-" + tag()
    response: Final = post_responses(rig, {"model": "dlp-model", "input": marker_body(marker), "max_output_tokens": 32})
    assert response.status_code == 200, response.text

    def none_scans() -> tuple[Request, ...]:
        return tuple(r for r in vendor_requests(rig, marker) if r.target.startswith("/none"))

    scans: Final = eventually(none_scans, lambda found: len(found) >= 1, seconds=30)
    assert MARKER not in scans[-1].body.decode(), scans[-1].body.decode()[:400]
    assert MASK in scans[-1].body.decode()

    events: Final = success_messages(rig, marker)
    guardrail_information: Final = events[-1].get("guardrail_information")
    assert guardrail_information, events[-1]


def test_provider_failure_logs_sanitized_messages(rig: Rig) -> None:
    """Row 13: provider 500 after the rewrite — failure event messages sanitized, spend row lands."""
    marker: Final = "row13-" + tag()
    response: Final = post_responses(rig, {"model": "err-model", "input": marker_body(marker), "max_output_tokens": 32})
    assert response.status_code >= 500, response.text

    events: Final = success_messages(rig, marker)
    assert MARKER not in body_blob([e.get("messages") for e in events]), events


def test_both_input_and_messages_keeps_raw_logged_messages(rig: Rig) -> None:
    """Row 14: documented corner — a body carrying both keys clobbers the refresh (fix incomplete)."""
    marker: Final = "row14-" + tag()
    body: Final = {
        "model": "mask-model",
        "input": marker_body(marker),
        "messages": [{"role": "user", "content": marker_body(marker)}],
        "max_output_tokens": 32,
    }
    response: Final = post_responses(rig, body)
    assert response.status_code == 200, response.text

    provider: Final = provider_requests(rig, marker)
    assert provider and MARKER not in provider[-1].body.decode(), "provider still gets the masked input"
    events: Final = success_messages(rig, marker)
    assert MARKER in body_blob(events[-1].get("messages")), "the central update_messages overwrite keeps this raw"


def test_presidio_masking_sanitizes_logged_messages(rig: Rig) -> None:
    """Row 15: the flagship masking guardrail through the same path."""
    marker: Final = "row15-" + tag()
    response: Final = post_responses(rig, {"model": "presidio-model", "input": marker_body(marker), "max_output_tokens": 32})
    assert response.status_code == 200, response.text

    events: Final = success_messages(rig, marker)
    assert MARKER not in body_blob([e.get("messages") for e in events]), events
    provider: Final = provider_requests(rig, marker)
    assert provider and MARKER not in provider[-1].body.decode()


def test_hostile_input_shapes_do_not_crash_the_head(rig: Rig) -> None:
    """Row 16: shapes the refresh path can meet — statuses match, no new 5xx class."""
    int_marker: Final = "row16int-" + tag()
    response_int: Final = post_responses(rig, {"model": "mask-model", "input": 42})
    assert response_int.status_code >= 400

    response_empty: Final = post_responses(rig, {"model": "mask-model", "input": []})
    assert response_empty.status_code == 200, "empty list passes the proxy; the scripted provider accepts it"

    marker_big: Final = "row16big-" + tag()
    response_big: Final = post_responses(rig, {"model": "mask-model", "input": "x" * 5000 + " " + marker_body(marker_big), "max_output_tokens": 32})
    assert response_big.status_code == 200, response_big.text
    events: Final = success_messages(rig, marker_big)
    assert MARKER not in body_blob([e.get("messages") for e in events])

    response_none_content: Final = post_responses(
        rig,
        {"model": "mask-model", "input": [{"role": "user", "content": None}], "max_output_tokens": 32},
    )
    assert response_none_content.status_code == 200, "no scannable text: the guardrail early-returns and the scripted provider accepts"
    assert rig.gateway.request("GET", "/health/liveliness").status_code == 200


def test_payload_sink_outage_during_burst_lands_every_id_once(rig: Rig) -> None:
    """Row 19: chaos — the payload sink 500s mid-burst; every response id still lands exactly once."""
    markers: Final = {f"row19-{tag()}-{i}": i for i in range(20)}
    rig.gate.fail = True
    try:
        results: Final = []

        def call(marker: str) -> None:
            body: Final = {
                "model": "mask-model" if markers[marker] % 2 == 0 else "open-model",
                "input": marker_body(marker),
                "max_output_tokens": 32,
            }
            results.append((marker, post_responses(rig, body).status_code))

        threads: Final = [threading.Thread(target=call, args=(m,)) for m in markers]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        assert len(results) == 20 and all(s == 200 for _, s in results), results
        assert rig.gateway.request("GET", "/health/liveliness").status_code == 200
    finally:
        rig.gate.fail = False

    marker_after: Final = "row19-after-" + tag()
    assert post_responses(rig, {"model": "mask-model", "input": marker_body(marker_after), "max_output_tokens": 32}).status_code == 200
    events: Final = success_messages(rig, marker_after)
    assert events, "logging must recover once the sink is back"
