"""Scripted sink peers, proxy rig and decoders shared by the dec1 logging sink contract tests.

One rig is one owned single-worker proxy wired to the dec1 scripted provider (`dec1_callbacks_support`) and to one
owned wire server per sink: Langfuse (OTLP v4 protobuf under `/api/public/otel/v1/traces`), a generic OTLP/HTTP
collector, the Datadog logs and LLM Observability intakes (both behind `DD_BASE_URL`), the `generic_api` webhook, a
Slack incoming webhook and an SQS queue URL. Every peer records each request it receives and can be switched to
answer 500, or to hold every request until released, so a sink outage can be scripted without touching the proxy.

Nothing here calls product code to compute an expected value. Decoders only turn wire bytes into dicts.
"""

from __future__ import annotations

import gzip
import json
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs

import httpx
from integration._support.client import Gateway, eventually, gateway_from_environment, held
from integration._support.otlp_sink import Span, decode_spans
from integration._support.process import owned_proxy
from integration._support.timing import assert_not_delayed_by
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.observability.dec1_callbacks_support import CHAT, ScriptedProvider, write_config
from pydantic import JsonValue

LANGFUSE_PUBLIC_KEY: Final = "pk-lf-dec1-sinks"
LANGFUSE_SECRET_KEY: Final = "sk-lf-dec1-sinks"
LANGFUSE_TRACES_PATH: Final = "/api/public/otel/v1/traces"
LANGFUSE_PROJECTS_PATH: Final = "/api/public/projects"
OTLP_TRACES_PATH: Final = "/v1/traces"
DD_API_KEY: Final = "synthetic-dd-api-key"
DD_SITE: Final = "us5.datadoghq.com"
DD_LOGS_PATH: Final = "/api/v2/logs"
DD_LLM_OBS_PATH: Final = "/api/intake/llm-obs/v1/trace/spans"
GENERIC_SECRET: Final = "synthetic-dec1-sinks-generic-secret"
SQS_QUEUE_PATH: Final = "/000000000000/dec1-sinks-queue"
SQS_ACCESS_KEY: Final = "AKIADEC1SINKSSYNTH"
SQS_SECRET_KEY: Final = "synthetic-dec1-sinks-sqs-secret"
REDACTED: Final = "redacted-by-litellm"
SINK_NAMES: Final = ("langfuse", "otlp", "datadog", "generic", "slack", "sqs")
STALL_SECONDS: Final = 30.0


@dataclass
class Switch:
    """How a peer answers: `status` for every request, and `hold` cleared to stall every request until set."""

    status: int = 200
    hold: threading.Event = field(default_factory=threading.Event)

    def __post_init__(self) -> None:
        self.hold.set()

    def stall(self) -> None:
        self.hold.clear()

    def release(self) -> None:
        self.hold.set()


@dataclass
class Peer:
    """A scripted sink: every request it received, kept across drains, and the switch steering its answers."""

    name: str
    switch: Switch
    wire: Wire | None = None
    seen: list[Request] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def url(self) -> str:
        assert self.wire is not None
        return self.wire.url

    def requests(self) -> tuple[Request, ...]:
        assert self.wire is not None
        with self.lock:
            self.seen.extend(self.wire.drain())
            return tuple(self.seen)

    def posts(self, path: str | None = None) -> tuple[Request, ...]:
        return tuple(
            request
            for request in self.requests()
            if request.method in ("POST", "PUT") and (path is None or request.target.split("?")[0] == path)
        )


def _projects() -> bytes:
    return json.dumps(
        {"data": [{"id": "dec1-sinks-project", "name": "dec1", "organization": {"id": "dec1-org", "name": "dec1"}}]}
    ).encode()


def _responder(peer: Peer) -> object:
    def respond(request: Request) -> Reply:
        if peer.name == "langfuse" and request.method == "GET" and request.target.startswith(LANGFUSE_PROJECTS_PATH):
            return Reply(body=_projects())
        peer.switch.hold.wait(timeout=STALL_SECONDS)
        if request.target.startswith("/always-500"):
            return Reply(status=500, body=b'{"error":"dec1 scripted 500"}')
        if request.target.startswith("/always-400"):
            return Reply(status=400, body=b'{"error":"dec1 scripted 400"}')
        if peer.switch.status != 200:
            return Reply(status=peer.switch.status, body=b'{"error":"dec1 scripted sink outage"}')
        if peer.name == "datadog":
            return Reply(status=202, body=b"{}")
        if peer.name in ("langfuse", "otlp"):
            return Reply(body=b"", content_type="application/x-protobuf")
        if peer.name == "slack":
            return Reply(body=b"ok", content_type="text/plain")
        if peer.name == "sqs":
            return Reply(
                body=b"<SendMessageResponse><SendMessageResult><MessageId>dec1</MessageId></SendMessageResult>"
                b"</SendMessageResponse>",
                content_type="text/xml",
            )
        return Reply()

    return respond


@dataclass(frozen=True)
class SinkRig:
    proxy: Gateway
    provider: Wire
    peers: Mapping[str, Peer]

    def peer(self, name: str) -> Peer:
        return self.peers[name]

    # ---- Langfuse and OTLP ----
    def spans(self, name: str = "langfuse", path: str | None = None) -> tuple[Span, ...]:
        """Every span the Langfuse peer (or the generic OTLP peer) received, decoded from protobuf or JSON."""
        found: list[Span] = []
        for request in self.peer(name).posts(path):
            found.extend(decode_spans(request.body, request.headers.get("content-type", "")))
        return tuple(found)

    def spans_where(
        self, name: str, attribute: str, value: JsonValue, *, count: int = 1, seconds: float = 30
    ) -> tuple[Span, ...]:
        """Wait until at least `count` spans carry `attribute == value`, then return all of them."""
        return eventually(
            lambda: tuple(span for span in self.spans(name) if span["attributes"].get(attribute) == value),
            lambda spans: len(spans) >= count,
            seconds=seconds,
        )

    def generation(self, call_id: str, *, seconds: float = 30) -> Span:
        """The single Langfuse generation span whose metadata names this x-litellm-call-id."""
        found: Final = self.spans_where(
            "langfuse", "langfuse.observation.metadata.litellm_call_id", call_id, seconds=seconds
        )
        generations: Final = tuple(
            span for span in found if span["attributes"].get("langfuse.observation.type") == "generation"
        )
        assert len(generations) == 1, [span["name"] for span in found]
        return generations[0]

    def otel_trace(self, call_id: str, *, seconds: float = 30) -> tuple[Span, ...]:
        """All OTLP spans in the trace whose `litellm_request` span carries `litellm.call_id == call_id`."""
        anchor: Final = self.spans_where("otlp", "litellm.call_id", call_id, seconds=seconds)[0]
        return eventually(
            lambda: tuple(span for span in self.spans("otlp") if span["trace_id"] == anchor["trace_id"]),
            lambda spans: any(span["name"] == "Received Proxy Server Request" for span in spans),
            seconds=seconds,
        )

    # ---- Datadog ----
    def datadog_logs(self) -> tuple[dict[str, JsonValue], ...]:
        """Every log entry posted to /api/v2/logs, gunzipped and flattened out of their batches."""
        entries: list[dict[str, JsonValue]] = []
        for request in self.peer("datadog").posts(DD_LOGS_PATH):
            body = request.body
            if request.headers.get("content-encoding") == "gzip":
                body = gzip.decompress(body)
            entries.extend(json.loads(body))
        return tuple(entries)

    def datadog_log(self, call_id: str, status: str | None = None, *, seconds: float = 30) -> dict[str, JsonValue]:
        """The single Datadog log whose `message` (a JSON StandardLoggingPayload) has this litellm_call_id."""

        def matching() -> tuple[dict[str, JsonValue], ...]:
            return tuple(
                entry
                for entry in self.datadog_logs()
                if call_id in str(entry.get("message")) and (status is None or entry.get("status") == status)
            )

        found: Final = eventually(matching, lambda entries: len(entries) >= 1, seconds=seconds)
        assert len(found) == 1, found
        return found[0]

    def llm_obs_batches(self) -> tuple[dict[str, JsonValue], ...]:
        return tuple(json.loads(request.body) for request in self.peer("datadog").posts(DD_LLM_OBS_PATH))

    def llm_obs_span(self, text: str, *, seconds: float = 30) -> dict[str, JsonValue]:
        """The single LLM Observability span whose JSON mentions `text` (a unique marker in the request)."""

        def matching() -> tuple[dict[str, JsonValue], ...]:
            return tuple(
                span
                for batch in self.llm_obs_batches()
                for span in batch["data"]["attributes"]["spans"]  # type: ignore[index]
                if text in json.dumps(span)
            )

        found: Final = eventually(matching, lambda spans: len(spans) >= 1, seconds=seconds)
        assert len(found) == 1, found
        return found[0]

    # ---- generic_api, Slack, SQS ----
    def generic_bodies(self) -> tuple[bytes, ...]:
        return tuple(request.body for request in self.peer("generic").posts())

    def slack_texts(self) -> tuple[str, ...]:
        return tuple(str(json.loads(request.body)["text"]) for request in self.peer("slack").posts())

    def slack_alert(self, needle: str, *, seconds: float = 60) -> str:
        found: Final = eventually(
            lambda: tuple(text for text in self.slack_texts() if needle in text),
            lambda texts: len(texts) >= 1,
            seconds=seconds,
        )
        return found[0]

    def sqs_messages(self) -> tuple[dict[str, JsonValue], ...]:
        """Decoded MessageBody of every SendMessage the SQS peer received."""
        messages: list[dict[str, JsonValue]] = []
        for request in self.peer("sqs").posts(SQS_QUEUE_PATH):
            form = parse_qs(request.body.decode())
            messages.append(json.loads(form["MessageBody"][0]))
        return tuple(messages)


@contextmanager
def sinks_rig(
    directory: Path,
    litellm_settings: Mapping[str, JsonValue],
    *,
    environment: Mapping[str, str] | None = None,
    general_settings: Mapping[str, JsonValue] | None = None,
    router_settings: Mapping[str, JsonValue] | None = None,
    sections: Mapping[str, JsonValue] | None = None,
    late: Callable[[Mapping[str, str]], Mapping[str, Mapping[str, JsonValue]]] | None = None,
) -> Iterator[SinkRig]:
    """Boot one single-worker proxy on `litellm_settings` with every sink peer's address in its environment.

    Only the callbacks named in `litellm_settings` are active; the environment points every sink integration at
    its peer so any of them can be turned on per test file. Flush intervals are cut to one second. `late` receives
    the peer URLs by sink name once the peers listen and returns extra "litellm_settings", "general_settings" or
    "sections" entries, for settings that must name a peer URL (named generic_api endpoints, per alert type
    webhooks).
    """
    peers: Final = {name: Peer(name, Switch()) for name in SINK_NAMES}
    with (
        wire_server(ScriptedProvider()) as provider,
        wire_server(_responder(peers["langfuse"])) as langfuse,  # type: ignore[arg-type]
        wire_server(_responder(peers["otlp"])) as otlp,  # type: ignore[arg-type]
        wire_server(_responder(peers["datadog"])) as datadog,  # type: ignore[arg-type]
        wire_server(_responder(peers["generic"])) as generic,  # type: ignore[arg-type]
        wire_server(_responder(peers["slack"])) as slack,  # type: ignore[arg-type]
        wire_server(_responder(peers["sqs"])) as sqs,  # type: ignore[arg-type]
        gateway_from_environment() as shared,
    ):
        for name, wire in zip(SINK_NAMES, (langfuse, otlp, datadog, generic, slack, sqs), strict=True):
            peers[name].wire = wire
        extra: Final = late({name: peer.url for name, peer in peers.items()}) if late else {}
        config: Final = write_config(
            directory / f"dec1-sinks-{uuid.uuid4().hex}.yaml",
            provider.url,
            {**litellm_settings, **extra.get("litellm_settings", {})},
            general_settings={**(general_settings or {}), **extra.get("general_settings", {})},
            router_settings=router_settings,
            sections={**(sections or {}), **extra.get("sections", {})},
        )
        overrides: Final = {
            "LANGFUSE_HOST": langfuse.url,
            "LANGFUSE_PUBLIC_KEY": LANGFUSE_PUBLIC_KEY,
            "LANGFUSE_SECRET_KEY": LANGFUSE_SECRET_KEY,
            "LANGFUSE_FLUSH_INTERVAL": "1",
            "OTEL_EXPORTER": "otlp_http",
            "OTEL_ENDPOINT": otlp.url + OTLP_TRACES_PATH,
            "OTEL_BSP_SCHEDULE_DELAY": "500",
            "OTEL_SERVICE_NAME": "dec1-sinks",
            "OTEL_ENVIRONMENT_NAME": "dec1-env",
            "DD_API_KEY": DD_API_KEY,
            "DD_SITE": DD_SITE,
            "DD_BASE_URL": datadog.url,
            "DD_SERVICE": "dec1-sinks-service",
            "DD_ENV": "dec1-env",
            "DD_SOURCE": "dec1-source",
            "GENERIC_LOGGER_ENDPOINT": generic.url,
            "GENERIC_LOGGER_HEADERS": f"Authorization=Bearer {GENERIC_SECRET}",
            "SLACK_WEBHOOK_URL": slack.url,
            "SQS_QUEUE_URL": sqs.url + SQS_QUEUE_PATH,
            "AWS_ACCESS_KEY_ID": SQS_ACCESS_KEY,
            "AWS_SECRET_ACCESS_KEY": SQS_SECRET_KEY,
            "DEFAULT_SQS_FLUSH_INTERVAL_SECONDS": "1",
            "DEFAULT_FLUSH_INTERVAL_SECONDS": "1",
            "STORE_MODEL_IN_DB": "False",
            **(environment or {}),
        }
        with owned_proxy(shared, directory, overrides, config=config) as proxy:
            yield SinkRig(proxy, provider, peers)


def new_call_id() -> str:
    return str(uuid.uuid4())


def marker() -> str:
    return "dec1sinks" + uuid.uuid4().hex


def chat(
    rig: SinkRig,
    call_id: str,
    text: str = "dec1 sink contract",
    *,
    model: str = CHAT,
    key: str | None = None,
    headers: Mapping[str, str] | None = None,
    **extra: JsonValue,
) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {"model": model, "messages": [{"role": "user", "content": text}], **extra}
    return rig.proxy.request(
        "POST", "/v1/chat/completions", body, key=key, headers={"x-litellm-call-id": call_id, **(headers or {})}
    )


def stream_chat(rig: SinkRig, call_id: str, text: str, *, key: str | None = None) -> tuple[int, str]:
    with rig.proxy.client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": CHAT, "messages": [{"role": "user", "content": text}], "stream": True},
        headers={"Authorization": f"Bearer {key or rig.proxy.key}", "x-litellm-call-id": call_id},
    ) as response:
        return response.status_code, response.read().decode()


def timed_chat(rig: SinkRig, call_id: str, text: str, **extra: JsonValue) -> tuple[httpx.Response, float]:
    """One chat request and its wall-clock duration in seconds, measured around the HTTP round trip only."""
    started: Final = time.monotonic()
    response: Final = chat(rig, call_id, text, **extra)
    return response, time.monotonic() - started


def healthy_seconds(rig: SinkRig) -> float:
    """Duration of one chat made while every sink is healthy, the control an outage call is compared against."""
    response, elapsed = timed_chat(rig, new_call_id(), marker())
    assert response.status_code == 200, response.text
    return elapsed


def assert_not_held(elapsed: float, control: float) -> None:
    """A call made during a sink outage must not wait for the sink: it may not take as long as the healthy control
    plus the STALL_SECONDS a stalled peer holds each request."""
    assert_not_delayed_by([elapsed], control=[control], injected_delay=STALL_SECONDS)


def settled(read: object, count: int, *, holding: float = 3) -> object:
    """`read()` once its length is `count` and stays `count` for `holding` seconds."""
    return held(read, lambda values: len(values) == count, holding=holding, seconds=30)  # type: ignore[arg-type]


def key_logging(callback_name: str, callback_vars: Mapping[str, JsonValue], callback_type: str = "success_and_failure"):
    """The `metadata.logging` entry a key or team carries to add one callback for its own requests."""
    return [{"callback_name": callback_name, "callback_type": callback_type, "callback_vars": dict(callback_vars)}]


def attribute(span: Span, name: str) -> JsonValue:
    return span["attributes"].get(name)


def parsed(value: JsonValue) -> JsonValue:
    """An attribute serialized as JSON text, decoded."""
    assert isinstance(value, str), value
    return json.loads(value)


def without(spans: Sequence[Span], call_id: str) -> tuple[Span, ...]:
    return tuple(span for span in spans if call_id not in json.dumps(span["attributes"]))
