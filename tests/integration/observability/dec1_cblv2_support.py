"""Scripted provider, scripted S3 peer, probe sink, config writer and readers for the dec1 cblv2 contract tests.

One rig is one owned single-worker proxy with three owned wire servers: the provider every deployment points at,
a scripted S3 peer that the `s3_v2` callback (now the Rust-backed logger, #51539) PUTs objects to, and a probe
sink the callback module `dec1_cblv2_probe` reports its hook calls to. Each test module boots its own rig with
the `litellm_settings` it needs (Prometheus, S3, ordered legacy loggers or V2 loggers).

Every deployment carries custom prices of 0.001 per input token and 0.002 per output token, and the provider
answers chat with 11 prompt and 4 completion tokens, so a chat costs exactly 0.019.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment, held
from integration._support.process import owned_proxy
from integration._support.prometheus_series import Sample, scrape
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

PROBE_MODULE: Final = "integration.observability.dec1_cblv2_probe"
PROVIDER_KEY: Final = "synthetic-cblv2-provider-key"
INPUT_RATE: Final = 0.001
OUTPUT_RATE: Final = 0.002
CHAT_USAGE: Final = (11, 4)
CHAT_COST: Final = 11 * INPUT_RATE + 4 * OUTPUT_RATE
EMBEDDING_TOKENS: Final = 5
REPLY_TEXT: Final = "cblv2 scripted reply"
PROVIDER_REJECTION: Final = "cblv2 scripted provider rejection"
PROVIDER_LIMIT: Final = "cblv2 scripted provider rate limit"
PROVIDER_OUTAGE: Final = "cblv2 scripted provider outage"
PROVIDER_DENIAL: Final = "cblv2 scripted provider denial"
PROVIDER_REMAINING_REQUESTS: Final = "97"
PROVIDER_REMAINING_TOKENS: Final = "9000"

CHAT: Final = "cblv2-chat"
CHAT_REJECTED: Final = "cblv2-chat-rejected"
CHAT_LIMITED: Final = "cblv2-chat-limited"
CHAT_OUTAGE: Final = "cblv2-chat-outage"
CHAT_DENIED: Final = "cblv2-chat-denied"
CHAT_FALLBACK_PRIMARY: Final = "cblv2-chat-fallback-primary"
CHAT_RPM_LIMITED: Final = "cblv2-chat-rpm-limited"
EMBEDDING: Final = "cblv2-embedding"
UPSTREAM_MODEL: Final = "gpt-4o-mini"

S3_BUCKET: Final = "cblv2-bucket"
S3_PATH: Final = "cblv2-logs"
S3_ACCESS_KEY: Final = "AKIAIOSFODNN7EXAMPLE"
S3_SECRET_KEY: Final = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"


def _chat_reply(identity: str, text: str) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "chat.completion",
            "created": 1,
            "model": UPSTREAM_MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": CHAT_USAGE[0], "completion_tokens": CHAT_USAGE[1], "total_tokens": 15},
        }
    ).encode()


def _chat_frames(identity: str, text: str) -> tuple[bytes, ...]:
    first, second = text[: len(text) // 2], text[len(text) // 2 :]
    head: Final = {"id": identity, "object": "chat.completion.chunk", "created": 1, "model": UPSTREAM_MODEL}
    chunks: Final = (
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": first}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {"content": second}, "finish_reason": None}]},
        {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": CHAT_USAGE[0], "completion_tokens": CHAT_USAGE[1], "total_tokens": 15},
        },
    )
    return (*(f"data: {json.dumps({**head, **chunk})}\n\n".encode() for chunk in chunks), b"data: [DONE]\n\n")


def _error(status: int, message: str, kind: str) -> Reply:
    return Reply(status=status, body=json.dumps({"error": {"message": message, "type": kind, "code": None}}).encode())


def scripted_provider(request: Request) -> Reply:
    """`/rejected/` answers 400, `/limited/` 429, `/outage/` 500 and `/denied/` 401. Successful chat replies carry
    the OpenAI `x-ratelimit-remaining-*` headers so the provider remaining gauges have a value to export."""
    if request.method == "GET":
        return Reply(body=b'{"object":"list","data":[]}')
    if request.target.startswith("/rejected/"):
        return _error(400, PROVIDER_REJECTION, "invalid_request_error")
    if request.target.startswith("/limited/"):
        return _error(429, PROVIDER_LIMIT, "rate_limit_error")
    if request.target.startswith("/outage/"):
        return _error(500, PROVIDER_OUTAGE, "server_error")
    if request.target.startswith("/denied/"):
        return _error(401, PROVIDER_DENIAL, "authentication_error")
    body: Final = json.loads(request.body or b"{}")
    identity: Final = "chatcmpl-cblv2-" + uuid.uuid4().hex
    limits: Final = MappingProxyType(
        {
            "x-ratelimit-remaining-requests": PROVIDER_REMAINING_REQUESTS,
            "x-ratelimit-remaining-tokens": PROVIDER_REMAINING_TOKENS,
        }
    )
    if request.target == "/v1/embeddings":
        return Reply(
            body=json.dumps(
                {
                    "object": "list",
                    "model": "text-embedding-3-small",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.5]}],
                    "usage": {"prompt_tokens": EMBEDDING_TOKENS, "total_tokens": EMBEDDING_TOKENS},
                }
            ).encode()
        )
    if body.get("stream") is True:
        return Reply(chunks=_chat_frames(identity, REPLY_TEXT), content_type="text/event-stream", headers=limits)
    return Reply(body=_chat_reply(identity, REPLY_TEXT), headers=limits)


@dataclass(frozen=True, slots=True)
class S3Put:
    target: str
    headers: Mapping[str, str]
    body: bytes

    @property
    def key(self) -> str:
        """Object key without the leading `/<bucket>/` of a path-style request."""
        return self.target.removeprefix(f"/{S3_BUCKET}/")

    def lines(self) -> tuple[dict[str, JsonValue], ...]:
        return tuple(json.loads(line) for line in self.body.splitlines() if line.strip())


@dataclass
class ScriptedS3:
    """Accepts every PUT with 200 and records it; `fail_next` answers that many PUTs with a 503 first."""

    fail_next: int = 0
    puts: list[S3Put] = field(default_factory=list)  # mutable-ok: appended under lock by the wire thread
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __call__(self, request: Request) -> Reply:
        if request.method != "PUT":
            return Reply(status=404, body=b"", content_type="application/xml")
        with self.lock:
            if self.fail_next > 0:
                self.fail_next -= 1
                return Reply(status=503, body=b"<Error><Code>SlowDown</Code></Error>", content_type="application/xml")
            self.puts.append(S3Put(request.target, dict(request.headers), request.body))
        return Reply(body=b"", content_type="application/xml")

    def recorded(self) -> tuple[S3Put, ...]:
        with self.lock:
            return tuple(self.puts)

    def objects_for(self, response_id: str) -> tuple[S3Put, ...]:
        return tuple(put for put in self.recorded() if any(line.get("id") == response_id for line in put.lines()))


@dataclass
class Journal:
    """Every JSON record the probe sink received, kept across drains because draining consumes the queue."""

    wire: Wire
    seen: list[dict[str, JsonValue]] = field(default_factory=list)  # mutable-ok: appended under lock
    lock: threading.Lock = field(default_factory=threading.Lock)

    def records(self) -> tuple[dict[str, JsonValue], ...]:
        with self.lock:
            self.seen.extend(json.loads(request.body) for request in self.wire.drain())
            return tuple(self.seen)


@dataclass(frozen=True)
class Rig:
    proxy: Gateway
    provider: Wire
    s3: ScriptedS3
    s3_wire: Wire
    probe: Journal

    def mentioning(self, text: str, logger: str | None = None) -> tuple[dict[str, JsonValue], ...]:
        """Probe records whose request text contains `text`, in the order the proxy process made the calls."""
        found: Final = tuple(
            record
            for record in self.probe.records()
            if text in str(record.get("text")) and (logger is None or record.get("logger") == logger)
        )
        return tuple(sorted(found, key=lambda record: int(str(record["seq"]))))

    def calls(self, text: str, logger: str | None = None) -> tuple[tuple[str, str], ...]:
        """(logger, hook) pairs for the request carrying `text`, in call order."""
        return tuple((str(record["logger"]), str(record["hook"])) for record in self.mentioning(text, logger))

    def settled_calls(
        self, text: str, count: int, logger: str | None = None, *, holding: float = 3
    ) -> tuple[tuple[str, str], ...]:
        """The (logger, hook) pairs once there are exactly `count` of them and no more arrive for `holding` s."""
        return held(lambda: self.calls(text, logger), lambda calls: len(calls) == count, holding=holding, seconds=30)

    def provider_texts(self) -> tuple[str, ...]:
        """Last user message of every chat request the provider received since the last read."""
        texts: list[str] = []
        for request in self.provider.drain():
            if request.method != "POST" or not request.body:
                continue
            messages = json.loads(request.body).get("messages") or [{}]
            texts.append(str(messages[-1].get("content")))
        return tuple(texts)


def _deployment(name: str, base: str, model: str = f"openai/{UPSTREAM_MODEL}", **extra: JsonValue) -> dict:
    return {
        "model_name": name,
        "litellm_params": {
            "model": model,
            "api_base": base,
            "api_key": PROVIDER_KEY,
            "input_cost_per_token": INPUT_RATE,
            "output_cost_per_token": OUTPUT_RATE,
            **extra,
        },
        "model_info": {"id": f"{name}-id"},
    }


def deployment_id(name: str) -> str:
    """Every deployment in the rig has the fixed model_info.id `<model_name>-id`, so model_id labels are literal."""
    return f"{name}-id"


def write_config(
    path: Path,
    provider_url: str,
    s3_url: str,
    litellm_settings: Mapping[str, JsonValue],
    *,
    s3_params: Mapping[str, JsonValue] | None = None,
    general_settings: Mapping[str, JsonValue] | None = None,
    router_settings: Mapping[str, JsonValue] | None = None,
) -> Path:
    settings: Final[dict[str, JsonValue]] = dict(litellm_settings)
    if s3_params is not None:
        settings["s3_callback_params"] = {
            "s3_bucket_name": S3_BUCKET,
            "s3_region_name": "us-east-1",
            "s3_endpoint_url": s3_url,
            "s3_path": S3_PATH,
            "s3_aws_access_key_id": S3_ACCESS_KEY,
            "s3_aws_secret_access_key": S3_SECRET_KEY,
            "s3_flush_interval": 1,
            **s3_params,
        }
    path.write_text(
        yaml.safe_dump(
            {
                "model_list": [
                    _deployment(CHAT, provider_url + "/v1"),
                    _deployment(CHAT_REJECTED, provider_url + "/rejected/v1"),
                    _deployment(CHAT_LIMITED, provider_url + "/limited/v1"),
                    _deployment(CHAT_OUTAGE, provider_url + "/outage/v1"),
                    _deployment(CHAT_DENIED, provider_url + "/denied/v1"),
                    _deployment(CHAT_FALLBACK_PRIMARY, provider_url + "/outage/v1"),
                    _deployment(CHAT_RPM_LIMITED, provider_url + "/v1", rpm=1),
                    _deployment(EMBEDDING, provider_url + "/v1", model="openai/text-embedding-3-small"),
                ],
                "litellm_settings": settings,
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "disable_model_info_refresh": True,
                    "proxy_batch_write_at": 1,
                    **(general_settings or {}),
                },
                "router_settings": {
                    "num_retries": 0,
                    "fallbacks": [{CHAT_FALLBACK_PRIMARY: [CHAT]}],
                    **(router_settings or {}),
                },
            }
        )
    )
    return path


@contextmanager
def cblv2_rig(
    directory: Path,
    litellm_settings: Mapping[str, JsonValue],
    *,
    s3_params: Mapping[str, JsonValue] | None = None,
    environment: Mapping[str, str] | None = None,
    general_settings: Mapping[str, JsonValue] | None = None,
    router_settings: Mapping[str, JsonValue] | None = None,
) -> Iterator[Rig]:
    """Boot one single-worker proxy on `litellm_settings`, wired to the scripted provider, S3 peer and probe."""
    s3: Final = ScriptedS3()
    with (
        wire_server(scripted_provider) as provider,
        wire_server(s3) as s3_wire,
        wire_server(lambda request: Reply()) as probe,
        gateway_from_environment() as shared,
    ):
        config: Final = write_config(
            directory / f"cblv2-{uuid.uuid4().hex}.yaml",
            provider.url,
            s3_wire.url,
            litellm_settings,
            s3_params=s3_params,
            general_settings=general_settings,
            router_settings=router_settings,
        )
        overrides: Final = {
            "CBLV2_PROBE_URL": probe.url,
            "STORE_MODEL_IN_DB": "False",
            **(environment or {}),
        }
        with owned_proxy(shared, directory, overrides, config=config) as proxy:
            yield Rig(proxy, provider, s3, s3_wire, Journal(probe))


def marker() -> str:
    return "cblv2" + uuid.uuid4().hex


def key_hash(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def chat(
    rig: Rig,
    text: str,
    *,
    model: str = CHAT,
    key: str | None = None,
    headers: Mapping[str, str] | None = None,
    **extra: JsonValue,
) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {"model": model, "messages": [{"role": "user", "content": text}], **extra}
    return rig.proxy.request("POST", "/v1/chat/completions", body, key=key, headers=dict(headers or {}))


def stream_chat(
    rig: Rig, text: str, *, model: str = CHAT, key: str | None = None, headers: Mapping[str, str] | None = None
) -> tuple[int, Mapping[str, str], str]:
    """Send a streamed chat request and read it to the end: status, response headers, raw SSE text."""
    with rig.proxy.client.stream(
        "POST",
        "/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": text}], "stream": True},
        headers={"Authorization": f"Bearer {key or rig.proxy.key}", **(headers or {})},
    ) as response:
        return response.status_code, dict(response.headers), response.read().decode()


def streamed_text(sse: str) -> str:
    payloads: Final = (line[6:] for line in sse.splitlines() if line.startswith("data: ") and line != "data: [DONE]")
    deltas: Final = (json.loads(payload).get("choices") for payload in payloads)
    return "".join(choices[0]["delta"].get("content") or "" for choices in deltas if choices)


def reply_text(response: httpx.Response) -> str:
    return str(response.json()["choices"][0]["message"]["content"])


def error_message(response: httpx.Response) -> str:
    return str(response.json()["error"]["message"])


def series(samples: Sequence[Sample], name: str, **labels: str) -> tuple[Sample, ...]:
    """Samples of one exposition name whose labels include every given label value."""
    return tuple(
        sample
        for sample in samples
        if sample.name == name and all(sample.labels.get(label) == value for label, value in labels.items())
    )


def values(samples: Sequence[Sample], name: str, **labels: str) -> list[float]:
    return [sample.value for sample in series(samples, name, **labels)]


def until_series(rig: Rig, name: str, *, seconds: float = 30, **labels: str) -> tuple[Sample, ...]:
    """Scrape /metrics until at least one sample of `name` carries the given labels."""
    return eventually(lambda: series(scrape(rig.proxy), name, **labels), bool, seconds=seconds)


def until_value(rig: Rig, name: str, expected: float, *, seconds: float = 30, **labels: str) -> tuple[Sample, ...]:
    """Scrape /metrics until the single matching series has exactly `expected` as its value."""
    return eventually(
        lambda: series(scrape(rig.proxy), name, **labels),
        lambda found: len(found) == 1 and found[0].value == expected,
        seconds=seconds,
    )


def label_names(sample: Sample) -> frozenset[str]:
    """Label names of a series, without the histogram `le` bucket attribute."""
    return frozenset(name for name in sample.labels if name != "le")
