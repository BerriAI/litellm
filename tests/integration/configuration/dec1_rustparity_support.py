"""Shared rig for the `test_dec1_rustparity_*` contracts: one config, one scripted provider, two process models.

Part A of the December 1 definition is "the Rust core in the Python proxy": with `LITELLM_RUST_FIRST=true` the
supervisor (litellm/proxy/rust_forward/supervisor) puts the Rust forwarder (crate
`litellm/rust/crates/gateway-forward`) on the public port and runs Python on a private loopback listener. The
forwarder's contract (gateway-forward/AGENTS.md) is that Python sees each request as if the client had connected
directly, and the client sees Python's answer unchanged except for hop-by-hop headers. These contracts check that
from the caller's side: the same request against the Python only proxy and against the Rust first proxy must give
the same status, the same stable headers, the same body, the same SSE framing, the same provider request and the
same `LiteLLM_SpendLogs` row.

How the rig boots (assumptions every file shares):

* Both proxies are started by `parity_rig` from one config file with `python -m integration._support.proxy`
  (the harness wrapper, cwd = the checkout, as `dec1_deploy_support.boot_outcome` does), one worker,
  `--use_prisma_db_push`, each with its own scratch Postgres and Redis from `process.private_state`. The only
  environment difference is `LITELLM_RUST_FIRST` (`""` versus `"true"`).
* `owned_proxy_process` is not used for the Rust side: its readiness probe looks for the listener inside the root
  process group, and the supervisor starts both children in their own sessions (rust_forward/platform/posix.py:34),
  so it would never see the Rust listener. Readiness here is `GET /health/readiness` answering 200.
* `parity_rig` asserts that the Rust side really is fronted by Rust: its log must not contain the supervisor's
  fallback notice (`SERVING_WITHOUT_RUST`) and the process listening on its public port must be the
  `litellm-gateway` binary. The configuration group is the one where CI installs that binary
  (`Install the Rust gateway` in test-integration.yml).
* Virtual keys are created on both proxies with the same raw value (`/key/generate` with `key`), so the hashed
  `api_key` in spend rows and the `x-litellm-key-*` headers are comparable.

Normalisation: values that differ on every call by design (call ids, generated object ids, timestamps, timing
headers) are removed before comparing; everything else must be equal. The volatile lists below are the whole list.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Final, Generic, TypeVar

import httpx
import psutil
import yaml
from pydantic import JsonValue

from litellm.proxy.rust_forward.supervisor import SERVING_WITHOUT_RUST
from tests.integration._support.client import JSON_OBJECT, Gateway, eventually
from tests.integration._support.ports import allocate_port
from tests.integration._support.process import private_state, stop_child
from tests.integration._support.vertex import service_account_json
from tests.integration._support.wire import Reply, Request, Wire, wire_server

T = TypeVar("T")

MASTER_KEY: Final = "sk-dec1-rustparity-master-not-a-secret"
PROVIDER_KEY: Final = "sk-dec1-rustparity-provider-not-a-secret"
AWS_ACCESS_KEY_ID: Final = "AKIDEC1RUSTPARITY000"
AWS_SECRET_ACCESS_KEY: Final = "dec1-rustparity-aws-secret-not-a-secret"
VERTEX_PROJECT: Final = "dec1-rustparity-project"
VERTEX_LOCATION: Final = "us-central1"
VERTEX_TOKEN: Final = "ya29.dec1-rustparity-token"
INPUT_COST: Final = 0.001
OUTPUT_COST: Final = 0.002
PROMPT_TOKENS: Final = 5
COMPLETION_TOKENS: Final = 3
EXPECTED_SPEND: Final = PROMPT_TOKENS * INPUT_COST + COMPLETION_TOKENS * OUTPUT_COST
REPLY_TEXT: Final = "dec1 rustparity reply"
PROVIDER_RETRY_AFTER: Final = "7"
BOOT_SECONDS: Final = 300
GATEWAY_BINARY: Final = "litellm-gateway"

CHAT: Final = "/v1/chat/completions"
MESSAGES: Final = "/v1/messages"
RESPONSES: Final = "/v1/responses"
EMBEDDINGS: Final = "/v1/embeddings"

OPENAI: Final = "rp-openai"
ANTHROPIC: Final = "rp-anthropic"
BEDROCK: Final = "rp-bedrock"
BEDROCK_INVOKE: Final = "rp-bedrock-invoke"
VERTEX: Final = "rp-vertex"
AZURE: Final = "rp-azure"
OPENAI_EMBED: Final = "rp-openai-embed"
AZURE_EMBED: Final = "rp-azure-embed"
BEDROCK_EMBED: Final = "rp-bedrock-embed"
VERTEX_EMBED: Final = "rp-vertex-embed"

VOLATILE_HEADERS: Final = frozenset(
    {
        "date",
        "content-length",
        "x-litellm-call-id",
        "x-litellm-response-duration-ms",
        "x-litellm-overhead-duration-ms",
        "x-litellm-callback-duration-ms",
        "x-litellm-key-spend",
        "x-request-id",
    }
)
"""Header values that legitimately differ between two calls; their names are still compared."""

VOLATILE_HEADER_PREFIXES: Final = ("x-litellm-timing-", "llm_provider-date", "llm_provider-x-request-id")
VOLATILE_FIELDS: Final = frozenset({"id", "created", "created_at", "response_id", "item_id", "system_fingerprint"})
"""Body fields generated per call (by the provider script or by litellm); dropped at any depth."""

HOP_BY_HOP: Final = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "proxy-connection",
    }
)
"""gateway-forward/src/rewrite.rs `HOP_BY_HOP`: the only headers Rust removes from Python's answer."""

SPEND_COLUMNS: Final = (
    "call_type",
    "api_key",
    "spend",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "model",
    "model_id",
    "model_group",
    "custom_llm_provider",
    "api_base",
    "user",
    "end_user",
    "team_id",
    "request_tags",
    "requester_ip_address",
    "status",
    "cache_hit",
)
"""The `LiteLLM_SpendLogs` columns compared between the two modes (schema.prisma:816)."""

_SPEND_SELECT: Final = ", ".join(f'"{column}"' for column in SPEND_COLUMNS)
"""Only the compared columns: `SELECT *` returns the timestamp columns as `datetime`, which `read_rows` refuses as
not JSON (`_support/database.py:21`)."""

_FAIL: Final = re.compile(r"fail=(\d{3})")
_ABORT: Final = "abort-mid-stream"
_HOLD: Final = "hold-at-provider"
_ENDLESS: Final = "endless-stream"


# --------------------------------------------------------------------------------------------- scripted provider


def _sse(event: str | None, payload: Mapping[str, JsonValue] | str) -> bytes:
    data: Final = payload if isinstance(payload, str) else json.dumps(payload)
    return (f"event: {event}\n" if event else "").encode() + f"data: {data}\n\n".encode()


def _reply_id() -> str:
    """A fresh suffix for every provider reply id. litellm keys `LiteLLM_SpendLogs.request_id` (the primary key) on
    the provider's response id (`get_spend_logs_id`, spend_tracking_utils.py:408-432), so a constant id would make
    every call after the first collide and lose its spend row."""
    return uuid.uuid4().hex[:16]


def _usage_openai() -> dict[str, JsonValue]:
    return {"prompt_tokens": PROMPT_TOKENS, "completion_tokens": COMPLETION_TOKENS, "total_tokens": 8}


def openai_chat(stream: bool) -> Reply:
    base: Final = {"id": f"chatcmpl-rustparity-{_reply_id()}", "created": 1, "model": "gpt-4o-mini"}
    if not stream:
        return Reply(
            body=json.dumps(
                {
                    **base,
                    "object": "chat.completion",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": REPLY_TEXT}, "finish_reason": "stop"}
                    ],
                    "usage": _usage_openai(),
                }
            ).encode()
        )
    chunk: Final = {**base, "object": "chat.completion.chunk"}
    return Reply(
        content_type="text/event-stream",
        chunks=(
            _sse(None, {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}),
            _sse(None, {**chunk, "choices": [{"index": 0, "delta": {"content": REPLY_TEXT}}]}),
            _sse(
                None,
                {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": _usage_openai()},
            ),
            b"data: [DONE]\n\n",
        ),
    )


def anthropic_message(stream: bool) -> Reply:
    message: Final[dict[str, JsonValue]] = {
        "id": f"msg_rustparity_{_reply_id()}",
        "type": "message",
        "role": "assistant",
        "model": "claude-rustparity",
        "content": [{"type": "text", "text": REPLY_TEXT}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": PROMPT_TOKENS, "output_tokens": COMPLETION_TOKENS},
    }
    if not stream:
        return Reply(body=json.dumps(message).encode())
    started: Final = {**message, "content": [], "stop_reason": None, "usage": {"input_tokens": 5, "output_tokens": 0}}
    return Reply(
        content_type="text/event-stream",
        chunks=(
            _sse("message_start", {"type": "message_start", "message": started}),
            _sse(
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            ),
            _sse(
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": REPLY_TEXT}},
            ),
            _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
            _sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": COMPLETION_TOKENS},
                },
            ),
            _sse("message_stop", {"type": "message_stop"}),
        ),
    )


def responses_object(stream: bool) -> Reply:
    done: Final[dict[str, JsonValue]] = {
        "id": f"resp_rustparity_{_reply_id()}",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": [
            {
                "id": "msg_resp_rustparity",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": REPLY_TEXT, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": PROMPT_TOKENS, "output_tokens": COMPLETION_TOKENS, "total_tokens": 8},
    }
    if not stream:
        return Reply(body=json.dumps(done).encode())
    return Reply(
        content_type="text/event-stream",
        chunks=(
            _sse(
                "response.created",
                {
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": {**done, "status": "in_progress", "output": []},
                },
            ),
            _sse(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "sequence_number": 1,
                    "item_id": "msg_resp_rustparity",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": REPLY_TEXT,
                },
            ),
            _sse("response.completed", {"type": "response.completed", "sequence_number": 2, "response": done}),
        ),
    )


def converse_reply() -> Reply:
    return Reply(
        body=json.dumps(
            {
                "output": {"message": {"role": "assistant", "content": [{"text": REPLY_TEXT}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": PROMPT_TOKENS, "outputTokens": COMPLETION_TOKENS, "totalTokens": 8},
                "metrics": {"latencyMs": 1},
            }
        ).encode()
    )


def gemini_reply(stream: bool) -> Reply:
    payload: Final[dict[str, JsonValue]] = {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": REPLY_TEXT}]}, "finishReason": "STOP", "index": 0}
        ],
        "usageMetadata": {
            "promptTokenCount": PROMPT_TOKENS,
            "candidatesTokenCount": COMPLETION_TOKENS,
            "totalTokenCount": 8,
        },
        "modelVersion": "gemini-2.0-flash",
    }
    if not stream:
        return Reply(body=json.dumps(payload).encode())
    return Reply(content_type="text/event-stream", chunks=(_sse(None, payload),))


def embedding_reply(target: str) -> Reply:
    vector: Final[list[JsonValue]] = [0.25, -0.5, 0.75]
    if "/model/" in target:
        return Reply(body=json.dumps({"embedding": vector, "inputTextTokenCount": PROMPT_TOKENS}).encode())
    if target.endswith(":predict"):
        return Reply(
            body=json.dumps(
                {
                    "predictions": [
                        {"embeddings": {"values": vector, "statistics": {"token_count": 5, "truncated": False}}}
                    ]
                }
            ).encode()
        )
    return Reply(
        body=json.dumps(
            {
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": vector}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": PROMPT_TOKENS, "total_tokens": PROMPT_TOKENS},
            }
        ).encode()
    )


def provider_error(target: str, status: int) -> Reply:
    """The error each provider family sends for `status`; a 429 also carries `retry-after: 7`."""
    headers: Final = {"retry-after": PROVIDER_RETRY_AFTER} if status == 429 else {}
    message: Final = f"scripted provider {status}"
    if target.startswith("/v1/messages"):
        kind = "rate_limit_error" if status == 429 else "api_error" if status >= 500 else "invalid_request_error"
        return Reply(
            status=status,
            headers=headers,
            body=json.dumps({"type": "error", "error": {"type": kind, "message": message}}).encode(),
        )
    if target.startswith("/model/"):
        return Reply(
            status=status,
            headers={**headers, "x-amzn-ErrorType": "ValidationException"},
            body=json.dumps({"message": message}).encode(),
        )
    if ":generateContent" in target or ":streamGenerateContent" in target or ":predict" in target:
        return Reply(
            status=status,
            headers=headers,
            body=json.dumps({"error": {"code": status, "message": message, "status": "INVALID_ARGUMENT"}}).encode(),
        )
    kind_openai: Final = (
        "rate_limit_error" if status == 429 else "server_error" if status >= 500 else "invalid_request_error"
    )
    return Reply(
        status=status,
        headers=headers,
        body=json.dumps({"error": {"message": message, "type": kind_openai, "code": None}}).encode(),
    )


def _streamed(request: Request) -> bool:
    if ":streamGenerateContent" in request.target or "converse-stream" in request.target:
        return True
    try:
        return JSON_OBJECT.validate_json(request.body or b"{}").get("stream") is True
    except ValueError:
        return False


def _success(request: Request) -> Reply:
    path: Final = request.target.split("?", 1)[0]
    stream: Final = _streamed(request)
    if path.endswith("/_oauth/token"):
        return Reply(
            body=json.dumps({"access_token": VERTEX_TOKEN, "expires_in": 3600, "token_type": "Bearer"}).encode()
        )
    if path.endswith(("/embeddings", ":predict")) or "titan-embed" in path:
        return embedding_reply(path)
    if path.endswith("/converse"):
        return converse_reply()
    if path.endswith(("/invoke", "/v1/messages")):
        return anthropic_message(stream)
    if ":generateContent" in path or ":streamGenerateContent" in path:
        return gemini_reply(stream)
    if path.endswith("/responses"):
        return responses_object(stream)
    if path.endswith("/chat/completions"):
        return openai_chat(stream)
    if path.endswith("/models"):
        return Reply(body=b'{"object": "list", "data": []}')
    return Reply(status=404, body=b'{"error": {"message": "scripted 404"}}')


@dataclass
class Provider:
    """One scripted peer that answers as OpenAI, Azure OpenAI, Anthropic, Bedrock (Converse and invoke) and
    Vertex AI (Gemini and embeddings). The prompt steers it: `fail=<status>` answers that status in the
    provider's own error shape, `abort-mid-stream` drops the connection after the first stream chunk,
    `hold-at-provider` sends the first chunk and waits for `release`, and `endless-stream` streams until the
    caller hangs up."""

    wire: Wire
    release: threading.Event
    arrived: threading.Event
    seen: list[Request] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def url(self) -> str:
        return self.wire.url

    def requests_for(self, case: str) -> tuple[Request, ...]:
        with self.lock:
            self.seen.extend(self.wire.drain())
            return tuple(request for request in self.seen if case.encode() in request.body)

    def hung_up(self) -> int:
        return self.wire.disconnected.qsize()


def _respond(release: threading.Event, arrived: threading.Event) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        text: Final = request.body.decode(errors="replace")
        failed: Final = _FAIL.search(text)
        if failed is not None:
            return provider_error(request.target.split("?", 1)[0], int(failed.group(1)))
        reply: Final = _success(request)
        if _ABORT in text and reply.chunks:
            return Reply(content_type=reply.content_type, chunks=reply.chunks, abort_after=1)
        if _HOLD in text:
            arrived.set()
            return Reply(
                content_type=reply.content_type, body=reply.body, chunks=reply.chunks, gate_after_first=release
            )
        if _ENDLESS in text and reply.chunks:
            return Reply(content_type=reply.content_type, chunks=reply.chunks[:1] * 600, pause_between_chunks=0.1)
        return reply

    return respond


@contextmanager
def scripted_provider() -> Iterator[Provider]:
    release: Final = threading.Event()
    arrived: Final = threading.Event()
    with wire_server(_respond(release, arrived)) as wire:
        try:
            yield Provider(wire, release, arrived)
        finally:
            release.set()


# --------------------------------------------------------------------------------------------- config


def _priced(**params: JsonValue) -> dict[str, JsonValue]:
    return {"input_cost_per_token": INPUT_COST, "output_cost_per_token": OUTPUT_COST, **params}


def model_list(url: str) -> list[dict[str, JsonValue]]:
    """Every deployment the matrix uses, all on the one scripted provider, all with explicit prices so spend is
    `5 * 0.001 + 3 * 0.002 = 0.011` for every chat-shaped call."""
    credentials: Final = service_account_json(VERTEX_PROJECT, url)
    aws: Final[dict[str, JsonValue]] = {
        "aws_access_key_id": AWS_ACCESS_KEY_ID,
        "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
        "aws_region_name": "us-east-1",
        "api_base": url,
    }
    vertex: Final[dict[str, JsonValue]] = {
        "vertex_project": VERTEX_PROJECT,
        "vertex_location": VERTEX_LOCATION,
        "vertex_credentials": credentials,
        "api_base": url,
    }
    azure: Final[dict[str, JsonValue]] = {"api_base": url, "api_key": PROVIDER_KEY, "api_version": "2024-10-21"}
    return [
        {
            "model_name": OPENAI,
            "litellm_params": _priced(model="openai/gpt-4o-mini", api_base=f"{url}/v1", api_key=PROVIDER_KEY),
        },
        {
            "model_name": ANTHROPIC,
            "litellm_params": _priced(model="anthropic/claude-rustparity", api_base=url, api_key=PROVIDER_KEY),
        },
        {
            "model_name": BEDROCK,
            "litellm_params": _priced(model="bedrock/converse/anthropic.claude-rustparity-v1", **aws),
        },
        {
            "model_name": BEDROCK_INVOKE,
            "litellm_params": _priced(model="bedrock/invoke/anthropic.claude-rustparity-v1", **aws),
        },
        {"model_name": VERTEX, "litellm_params": _priced(model="vertex_ai/gemini-2.0-flash", **vertex)},
        {"model_name": AZURE, "litellm_params": _priced(model="azure/rp-deployment", **azure)},
        {
            "model_name": OPENAI_EMBED,
            "litellm_params": _priced(
                model="openai/text-embedding-3-small", api_base=f"{url}/v1", api_key=PROVIDER_KEY
            ),
        },
        {"model_name": AZURE_EMBED, "litellm_params": _priced(model="azure/rp-embedding", **azure)},
        {"model_name": BEDROCK_EMBED, "litellm_params": _priced(model="bedrock/amazon.titan-embed-text-v2:0", **aws)},
        {"model_name": VERTEX_EMBED, "litellm_params": _priced(model="vertex_ai/text-embedding-004", **vertex)},
    ]


def parity_config(
    url: str,
    *,
    general_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    return {
        "model_list": model_list(url),
        "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY", **(general_settings or {})},
        "litellm_settings": dict(litellm_settings or {}),
        # Cooldowns off: one rig serves a whole module, and a scripted provider 429 or 5xx would otherwise park the
        # single deployment of a model group for 5 seconds and turn the next tests' calls into the router's own
        # `all_deployments_in_cooldown` 429 (litellm/router_utils/cooldown_handlers.py:414).
        "router_settings": {"num_retries": 0, "disable_cooldowns": True},
    }


# --------------------------------------------------------------------------------------------- proxies


@dataclass(frozen=True, slots=True)
class Booted:
    """One proxy of the pair: its client (as a harness `Gateway`, so `rows` reads its own database), its process
    and its log."""

    name: str
    gateway: Gateway
    process: subprocess.Popen[bytes]
    log: Path
    port: int

    def log_text(self) -> str:
        return self.log.read_text(errors="replace")


def _proxy_root() -> Path:
    return Path(os.environ.get("INTEGRATION_PROXY_ROOT") or Path(__file__).resolve().parents[3])


def _ready(port: int) -> bool:
    try:
        with httpx.Client(timeout=2, trust_env=False) as client:
            return client.get(f"http://127.0.0.1:{port}/health/readiness").status_code == 200
    except httpx.TransportError:
        return False


def listener_names(port: int) -> frozenset[str]:
    """Names of the processes listening on `port` on this host."""
    names: set[str] = set()
    for connection in psutil.net_connections(kind="inet"):
        if connection.status == psutil.CONN_LISTEN and connection.laddr.port == port and connection.pid:
            try:
                names.add(psutil.Process(connection.pid).name())
            except psutil.Error:
                continue
    return frozenset(names)


@contextmanager
def booted_proxy(
    name: str, config: Path, directory: Path, rust_first: bool, environment: Mapping[str, str]
) -> Iterator[Booted]:
    directory.mkdir(parents=True, exist_ok=True)
    port: Final = allocate_port()
    log_path: Final = directory / f"rustparity-{name}-{uuid.uuid4().hex}.log"
    with ExitStack() as stack:
        state: Final = stack.enter_context(private_state(directory, os.environ))
        env: Final = {
            **os.environ,
            **state.environment,
            "LITELLM_MASTER_KEY": MASTER_KEY,
            "LITELLM_SALT_KEY": os.environ.get("LITELLM_SALT_KEY", "sk-integration-salt"),
            "STORE_MODEL_IN_DB": "False",
            "PYTHON_DOTENV_DISABLED": "1",
            "LITELLM_RUST_FIRST": "true" if rust_first else "",
            **environment,
        }
        command: Final = (
            sys.executable,
            "-m",
            "integration._support.proxy",
            "--config",
            str(config),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--num_workers",
            "1",
            "--use_prisma_db_push",
        )
        with log_path.open("w") as log:
            process: Final = subprocess.Popen(  # test-quality-ok: the checkout is the proxy under test
                command, cwd=_proxy_root(), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        try:
            deadline: Final = time.monotonic() + BOOT_SECONDS
            while not _ready(port):
                assert process.poll() is None, log_path.read_text(errors="replace")[-4000:]
                assert time.monotonic() < deadline, f"{name} proxy never became ready"
                time.sleep(0.2)
            client: Final = stack.enter_context(
                httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60, trust_env=False)
            )
            yield Booted(name, Gateway(client, MASTER_KEY, "", state.database_url), process, log_path, port)
        finally:
            stop_child(process, grace_seconds=30)


@dataclass(frozen=True, slots=True)
class Pair(Generic[T]):
    """The same observation taken on the Python only proxy and on the Rust first proxy."""

    python: T
    rust: T

    def same(self) -> bool:
        return self.python == self.rust


@dataclass(frozen=True, slots=True)
class Seen:
    """What a caller saw for one request: status, headers (lower case names, repeated names joined), raw body,
    and the prompt marker that request carried."""

    status: int
    headers: Mapping[str, str]
    raw_headers: tuple[tuple[str, str], ...]
    body: bytes
    case: str

    def json(self) -> dict[str, JsonValue]:
        return JSON_OBJECT.validate_json(self.body)

    @property
    def call_id(self) -> str:
        return self.headers.get("x-litellm-call-id", "")


def seen_of(response: httpx.Response, body: bytes | None = None, case: str = "") -> Seen:
    return Seen(
        response.status_code,
        {name.lower(): value for name, value in response.headers.items()},
        tuple((name.lower(), value) for name, value in response.headers.multi_items()),
        response.content if body is None else body,
        case,
    )


BODY_FAILED: Final = b"\n<body failed>"
"""Appended by `read_until_failure` when the response body broke off instead of ending cleanly."""


def read_until_failure(response: httpx.Response) -> bytes:
    """Every body byte the caller received; a body that fails mid-way ends with `BODY_FAILED`."""
    pieces: Final[list[bytes]] = []
    try:
        for piece in response.iter_bytes():
            pieces.append(piece)
    except httpx.HTTPError:
        pieces.append(BODY_FAILED)
    return b"".join(pieces)


def marker() -> str:
    return f"rustparity-{uuid.uuid4().hex}"


@dataclass(frozen=True, slots=True)
class Rig:
    python: Booted
    rust: Booted
    provider: Provider

    def both(self, observe: Callable[[Booted], T]) -> Pair[T]:
        return Pair(observe(self.python), observe(self.rust))

    def send(
        self,
        path: str,
        body: Callable[[str], Mapping[str, JsonValue]],
        *,
        key: str | None = MASTER_KEY,
        headers: Mapping[str, str] | None = None,
        stream: bool = False,
    ) -> Pair[Seen]:
        """Post `body(marker)` to `path` on both proxies, each with its own marker, and read the whole answer."""

        def one(proxy: Booted) -> Seen:
            case: Final = marker()
            sent: Final = {**({"Authorization": f"Bearer {key}"} if key is not None else {}), **(headers or {})}
            if not stream:
                response: Final = proxy.gateway.client.post(path, json=dict(body(case)), headers=sent)
                return seen_of(response, response.content, case)
            with proxy.gateway.client.stream("POST", path, json=dict(body(case)), headers=sent) as streamed:
                return seen_of(streamed, read_until_failure(streamed), case)

        return self.both(one)

    def send_raw(self, path: str, content: bytes, headers: Mapping[str, str]) -> Pair[Seen]:
        """Post exact bytes (no marker substitution) to `path` on both proxies."""

        def one(proxy: Booted) -> Seen:
            response: Final = proxy.gateway.client.post(path, content=content, headers=dict(headers))
            return seen_of(response, response.content, "")

        return self.both(one)

    def same_key(self, **fields: JsonValue) -> str:
        """Create a virtual key with one raw value on both proxies and return it."""
        raw: Final = f"sk-rustparity-{uuid.uuid4().hex}"
        for proxy in (self.python, self.rust):
            created = proxy.gateway.post("/key/generate", {"key": raw, **fields})  # rebind-ok: one per proxy
            assert created["key"] == raw, created
        return raw

    def provider_requests(self, seen: Pair[Seen]) -> Pair[tuple[Request, ...]]:
        return Pair(self.provider.requests_for(seen.python.case), self.provider.requests_for(seen.rust.case))


@contextmanager
def parity_rig(
    directory: Path,
    *,
    general_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
    environment: Mapping[str, str] | None = None,
) -> Iterator[Rig]:
    """Boot the Python only proxy and the Rust first proxy on one config and one scripted provider."""
    with scripted_provider() as provider:
        config: Final = directory / "rustparity-config.yaml"
        directory.mkdir(parents=True, exist_ok=True)
        config.write_text(
            yaml.safe_dump(
                parity_config(provider.url, general_settings=general_settings, litellm_settings=litellm_settings),
                sort_keys=False,
            )
        )
        with (
            booted_proxy("python", config, directory / "python", False, environment or {}) as python,
            booted_proxy("rust", config, directory / "rust", True, environment or {}) as rust,
        ):
            assert SERVING_WITHOUT_RUST not in rust.log_text(), rust.log_text()[-4000:]
            assert GATEWAY_BINARY in listener_names(rust.port), listener_names(rust.port)
            assert GATEWAY_BINARY not in listener_names(python.port), listener_names(python.port)
            yield Rig(python, rust, provider)


# --------------------------------------------------------------------------------------------- normalisation


def without_volatile(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {key: without_volatile(item) for key, item in value.items() if key not in VOLATILE_FIELDS}
    if isinstance(value, list):
        return [without_volatile(item) for item in value]
    return value


def comparable_body(seen: Seen) -> JsonValue:
    return without_volatile(JSON_OBJECT.validate_json(seen.body.replace(seen.case.encode(), b"CASE")))


def comparable_error(seen: Seen) -> JsonValue:
    """An error body with the marker and the call id (some envelopes quote it) replaced by placeholders and
    per-call fields dropped; a body that is not JSON is kept as text."""
    text: Final = (
        seen.body.decode(errors="replace").replace(seen.case, "CASE")
        if seen.case
        else seen.body.decode(errors="replace")
    )
    cleaned: Final = text.replace(seen.call_id, "CALL_ID") if seen.call_id else text
    try:
        return without_volatile(json.loads(cleaned))
    except ValueError:
        return cleaned


def _volatile_header(name: str) -> bool:
    return name in VOLATILE_HEADERS or name.startswith(VOLATILE_HEADER_PREFIXES)


def stable_headers(seen: Seen) -> dict[str, str]:
    """Every response header except hop-by-hop ones, with volatile values blanked (their names are kept)."""
    return {
        name: ("<volatile>" if _volatile_header(name) else value)
        for name, value in seen.headers.items()
        if name not in HOP_BY_HOP
    }


def litellm_header_names(seen: Seen) -> frozenset[str]:
    return frozenset(name for name in seen.headers if name.startswith("x-litellm-"))


def sse_events(seen: Seen) -> tuple[tuple[str, JsonValue], ...]:
    """`(event name, normalised data)` for every SSE block; `[DONE]` is kept as the string `"[DONE]"`."""
    events: list[tuple[str, JsonValue]] = []
    text: Final = seen.body.decode(errors="replace").replace(seen.case, "CASE").replace("\r\n", "\n")
    for block in text.split("\n\n"):
        name = ""
        data = ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            events.append((name, "[DONE]"))
        elif data:
            events.append((name, without_volatile(json.loads(data))))
    return tuple(events)


def sse_lines_shape(seen: Seen) -> tuple[str, ...]:
    """The framing of an SSE body: the field name of every line (`event`, `data`, or empty for the separator)."""
    return tuple(line.split(":", 1)[0] for line in seen.body.decode(errors="replace").split("\n"))


def provider_body(request: Request, case: str) -> JsonValue:
    return without_volatile(JSON_OBJECT.validate_json(request.body.replace(case.encode(), b"CASE")))


PROVIDER_VOLATILE_HEADERS: Final = frozenset(
    {
        "content-length",
        "x-amz-date",
        "authorization",
        "x-amz-content-sha256",
        "amz-sdk-invocation-id",
        "x-litellm-call-id",
        "traceparent",
        "x-request-id",
        "host",
    }
)


def provider_header_names(request: Request) -> frozenset[str]:
    return frozenset(name.lower() for name in request.headers)


def stable_provider_headers(request: Request) -> dict[str, str]:
    return {
        name.lower(): value for name, value in request.headers.items() if name.lower() not in PROVIDER_VOLATILE_HEADERS
    }


def spend_row(proxy: Booted, call_id: str, seconds: float = 90) -> dict[str, JsonValue]:
    """The one spend row the proxy writes for `call_id`, polled until it lands, reduced to `SPEND_COLUMNS`."""
    rows: Final = eventually(
        lambda: proxy.gateway.rows(
            f'SELECT {_SPEND_SELECT} FROM "LiteLLM_SpendLogs" WHERE litellm_call_id = %s OR request_id = %s',
            (call_id, call_id),
        ),
        lambda found: len(found) >= 1,
        seconds=seconds,
    )
    assert len(rows) == 1, rows
    return {column: rows[0].get(column) for column in SPEND_COLUMNS}


def spend_rows_for(proxy: Booted, case: str) -> list[dict[str, JsonValue]]:
    """Spend rows whose stored prompt mentions `case`; used for refused calls that return no call id."""
    return proxy.gateway.rows(
        'SELECT request_id, litellm_call_id FROM "LiteLLM_SpendLogs" '
        "WHERE messages::text LIKE %s OR proxy_server_request::text LIKE %s",
        (f"%{case}%", f"%{case}%"),
    )


# --------------------------------------------------------------------------------------------- request bodies


def chat_body(model: str, **fields: JsonValue) -> Callable[[str], Mapping[str, JsonValue]]:
    return lambda case: {"model": model, "messages": [{"role": "user", "content": case}], **fields}


def messages_body(model: str, **fields: JsonValue) -> Callable[[str], Mapping[str, JsonValue]]:
    return lambda case: {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": case}], **fields}


def responses_body(model: str, **fields: JsonValue) -> Callable[[str], Mapping[str, JsonValue]]:
    return lambda case: {"model": model, "input": case, **fields}


def embeddings_body(model: str, **fields: JsonValue) -> Callable[[str], Mapping[str, JsonValue]]:
    return lambda case: {"model": model, "input": case, **fields}


def directive(build: Callable[[str], Mapping[str, JsonValue]], text: str) -> Callable[[str], Mapping[str, JsonValue]]:
    """`build` with the marker followed by a provider directive such as `fail=503`."""
    return lambda case: build(f"{case} {text}")


BODY_FOR_PATH: Final[Mapping[str, Callable[[str], Callable[[str], Mapping[str, JsonValue]]]]] = {
    CHAT: chat_body,
    MESSAGES: messages_body,
    RESPONSES: responses_body,
    EMBEDDINGS: embeddings_body,
}


@dataclass(frozen=True, slots=True)
class Cell:
    """One cell of the endpoint by provider matrix."""

    path: str
    model: str

    def body(self, **fields: JsonValue) -> Callable[[str], Mapping[str, JsonValue]]:
        return BODY_FOR_PATH[self.path](self.model, **fields)


UNARY_CELLS: Final[Mapping[str, Cell]] = {
    "chat-openai": Cell(CHAT, OPENAI),
    "chat-anthropic": Cell(CHAT, ANTHROPIC),
    "chat-bedrock": Cell(CHAT, BEDROCK),
    "chat-vertex": Cell(CHAT, VERTEX),
    "chat-azure": Cell(CHAT, AZURE),
    "messages-anthropic": Cell(MESSAGES, ANTHROPIC),
    "messages-bedrock": Cell(MESSAGES, BEDROCK_INVOKE),
    "messages-vertex": Cell(MESSAGES, VERTEX),
    "messages-azure": Cell(MESSAGES, AZURE),
    "messages-openai": Cell(MESSAGES, OPENAI),
    "responses-openai": Cell(RESPONSES, OPENAI),
    "responses-azure": Cell(RESPONSES, AZURE),
    "responses-anthropic": Cell(RESPONSES, ANTHROPIC),
    "responses-bedrock": Cell(RESPONSES, BEDROCK),
    "responses-vertex": Cell(RESPONSES, VERTEX),
    "embeddings-openai": Cell(EMBEDDINGS, OPENAI_EMBED),
    "embeddings-azure": Cell(EMBEDDINGS, AZURE_EMBED),
    "embeddings-bedrock": Cell(EMBEDDINGS, BEDROCK_EMBED),
    "embeddings-vertex": Cell(EMBEDDINGS, VERTEX_EMBED),
}

STREAM_CELLS: Final[Mapping[str, Cell]] = {
    "chat-openai": Cell(CHAT, OPENAI),
    "chat-anthropic": Cell(CHAT, ANTHROPIC),
    "chat-vertex": Cell(CHAT, VERTEX),
    "chat-azure": Cell(CHAT, AZURE),
    "messages-anthropic": Cell(MESSAGES, ANTHROPIC),
    "messages-openai": Cell(MESSAGES, OPENAI),
    "messages-azure": Cell(MESSAGES, AZURE),
    "responses-openai": Cell(RESPONSES, OPENAI),
    "responses-anthropic": Cell(RESPONSES, ANTHROPIC),
}
"""Bedrock streams are AWS event streams the scripted provider does not encode; see the index "Not covered"."""

GENERATION_CELLS: Final[Mapping[str, Cell]] = {
    name: cell for name, cell in UNARY_CELLS.items() if cell.path != EMBEDDINGS
}


# --------------------------------------------------------------------------------------------- raw HTTP


@dataclass(frozen=True, slots=True)
class RawAnswer:
    status: int
    headers: tuple[tuple[str, str], ...]
    body: bytes

    def header(self, name: str) -> str | None:
        return next((value for key, value in self.headers if key == name.lower()), None)


def read_answer(stream: BinaryIO) -> RawAnswer:
    status_line: Final = stream.readline().decode("latin-1")
    assert status_line.startswith("HTTP/1."), status_line
    status: Final = int(status_line.split(" ", 2)[1])
    headers: list[tuple[str, str]] = []
    while (line := stream.readline().decode("latin-1").rstrip("\r\n")) != "":
        name, _, value = line.partition(":")
        headers.append((name.strip().lower(), value.strip()))
    lookup: Final = dict(headers)
    if 100 <= status < 200 or status in (204, 304):
        # An interim or bodiless answer (RFC 9112 section 6.3): reading to EOF would block until the server closes,
        # so `100 Continue` from an `Expect` exchange returns at once.
        return RawAnswer(status, tuple(headers), b"")
    if lookup.get("transfer-encoding", "").lower() == "chunked":
        pieces: list[bytes] = []
        while (size := int(stream.readline().split(b";", 1)[0].strip() or b"0", 16)) > 0:
            pieces.append(stream.read(size))
            stream.readline()
        while stream.readline() not in (b"\r\n", b"\n", b""):
            pass
        return RawAnswer(status, tuple(headers), b"".join(pieces))
    if "content-length" in lookup:
        return RawAnswer(status, tuple(headers), stream.read(int(lookup["content-length"])))
    return RawAnswer(status, tuple(headers), stream.read())


def raw_exchange(proxy: Booted, payload: bytes, timeout: float = 30) -> RawAnswer:
    """Write `payload` on a fresh connection to `proxy` and read one answer."""
    with (
        socket.create_connection(("127.0.0.1", proxy.port), timeout=timeout) as client,
        client.makefile("rb") as stream,
    ):
        client.sendall(payload)
        return read_answer(stream)


def raw_request(
    method: str, target: str, headers: Sequence[tuple[str, str]], body: bytes = b"", version: str = "HTTP/1.1"
) -> bytes:
    head: Final = "".join(f"{name}: {value}\r\n" for name, value in headers)
    return f"{method} {target} {version}\r\n{head}\r\n".encode("latin-1") + body


def chat_bytes(model: str, text: str) -> bytes:
    return json.dumps({"model": model, "messages": [{"role": "user", "content": text}]}).encode()


def gzipped(data: bytes) -> bytes:
    return gzip.compress(data, mtime=0)
