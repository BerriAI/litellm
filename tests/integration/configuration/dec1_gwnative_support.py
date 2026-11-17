"""Shared rig for the `test_dec1_gwnative_*` contracts on the native Rust gateway preview.

The preview is the standalone `litellm-gateway` binary from the `litellm-proxy` package
(`litellm/packaging/litellm-proxy`, built from `litellm/rust/crates/gateway`). CI installs it into the
virtualenv for the management group only ("Install the Rust gateway" in
`.github/workflows/test-integration.yml`), which is why these files live in `configuration/`.

The binary runs standalone when `LITELLM_FORWARD_UPSTREAM` is absent (`gateway/src/main.rs`): it reads
`HOST` (default `0.0.0.0`), `PORT` (default `4000`), `LITELLM_CONFIG` (default `config.yaml`) and
`RUST_LOG`, loads the YAML with `os.environ/` references resolved, builds one deployment per
`model_name`, and serves the inference routes behind master-key-only authentication. It never opens a
database. `native_gateway` boots it on a private port with a config the test writes, waits for
`/health/readiness`, and on exit sends SIGTERM and requires exit status 0.

`ScriptedProvider` is one local HTTP peer that plays every provider at once. Each test puts a unique
marker in its prompt and scripts the reply for that marker, so the module can share one gateway and
still read back exactly the provider requests its own call caused.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import pytest
import yaml
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import eventually
from tests.integration._support.ports import allocate_port
from tests.integration._support.process import kill_stray_process, send_signal
from tests.integration._support.rust_forward import MASTER_KEY
from tests.integration._support.wire import Reply, Request, Wire, wire_server

GATEWAY_KEY: Final = MASTER_KEY
PROVIDER_KEY: Final = "sk-dec1-gwnative-provider-not-a-secret"
AWS_ACCESS_KEY_ID: Final = "AKIDEC1GWNATIVE00000"
AWS_SECRET_ACCESS_KEY: Final = "dec1-gwnative-aws-secret-not-a-secret"
VERTEX_PROJECT: Final = "dec1-gwnative-project"
VERTEX_LOCATION: Final = "us-east5"
VERTEX_TOKEN: Final = "ya29.dec1-gwnative-token"
BOOT_SECONDS: Final = 60
EXIT_SECONDS: Final = 15
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
# axum's limit in `gateway-inference/src/request.rs`: 50 MiB of file plus 1 MiB of form overhead.
MAX_BODY_BYTES: Final = 50 * 1024 * 1024 + 1024 * 1024
UNAUTHENTICATED: Final = "missing or invalid bearer token"
UNCONFIGURED: Final = "gateway auth not configured"
_DROPPED_ENVIRONMENT_PREFIXES: Final = ("LITELLM_FORWARD_", "DATABASE_", "AWS_", "AZURE_", "VERTEX")
_DROPPED_ENVIRONMENT: Final = frozenset({"LITELLM_UI_PATH", "DIRECT_URL", "LITELLM_CONFIG", "HOST", "PORT"})


def gateway_binary() -> Path:
    """The installed `litellm-gateway` next to the interpreter, or a skip when this venv has none."""
    binary: Final = Path(sys.executable).with_name("litellm-gateway")
    if not binary.exists():
        pytest.skip("litellm-gateway is not installed; CI installs it for the management group only")
    return binary


@dataclass(frozen=True, slots=True)
class NativeGateway:
    port: int
    process: subprocess.Popen[bytes]
    log: Path

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def client(self, timeout: float = 30) -> httpx.Client:
        return httpx.Client(base_url=self.url, timeout=timeout, trust_env=False)

    def post(
        self,
        path: str,
        body: Mapping[str, JsonValue] | None = None,
        *,
        key: str | None = GATEWAY_KEY,
        headers: Mapping[str, str] = MappingProxyType({}),
        content: bytes | None = None,
    ) -> httpx.Response:
        sent: Final = {**({"Authorization": f"Bearer {key}"} if key is not None else {}), **headers}
        with self.client() as client:
            if content is not None:
                return client.post(path, content=content, headers={"content-type": "application/json", **sent})
            return client.post(path, json=dict(body or {}), headers=sent)

    def get(self, path: str, *, key: str | None = None) -> httpx.Response:
        with self.client() as client:
            return client.get(path, headers={"Authorization": f"Bearer {key}"} if key is not None else {})

    def records(self) -> tuple[dict[str, JsonValue], ...]:
        """The JSON log lines the gateway wrote to stderr, in order (`StderrSink` in `gateway/src/main.rs`)."""
        lines: Final = self.log.read_text(errors="replace").splitlines()
        parsed: list[dict[str, JsonValue]] = []
        for line in lines:
            if line.startswith("{"):
                try:
                    parsed.append(JSON_OBJECT.validate_json(line))
                except ValueError:
                    continue
        return tuple(parsed)


def _environment(config: Path, port: int, environment: Mapping[str, str], master_key: str | None) -> dict[str, str]:
    inherited: Final = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(_DROPPED_ENVIRONMENT_PREFIXES)
        and name not in _DROPPED_ENVIRONMENT
        and not name.endswith("_API_KEY")
        and name != "LITELLM_MASTER_KEY"
    }
    return {
        **inherited,
        "HOST": "127.0.0.1",
        "PORT": str(port),
        "LITELLM_CONFIG": str(config),
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        **({"LITELLM_MASTER_KEY": master_key} if master_key is not None else {}),
        **environment,
    }


def write_config(directory: Path, config: Mapping[str, JsonValue], name: str = "config.yaml") -> Path:
    path: Final = directory / name
    path.write_text(yaml.safe_dump(json.loads(json.dumps(config)), sort_keys=False))
    return path


def _ready(url: str) -> bool:
    try:
        return httpx.get(f"{url}/health/readiness", timeout=2, trust_env=False).status_code == 200
    except httpx.TransportError:
        return False


def begin_shutdown(gateway: NativeGateway, signum: signal.Signals = signal.SIGTERM) -> None:
    """Send `signum` without waiting, for tests that keep a call open while the gateway drains."""
    send_signal(gateway.process, signum)


def terminate(gateway: NativeGateway, signum: signal.Signals = signal.SIGTERM) -> int | None:
    """Send `signum` and return the exit status, or None when the gateway is still running after EXIT_SECONDS."""
    send_signal(gateway.process, signum)
    try:
        return gateway.process.wait(timeout=EXIT_SECONDS)
    except subprocess.TimeoutExpired:
        return None


@contextmanager
def native_gateway(
    config: Mapping[str, JsonValue],
    directory: Path,
    environment: Mapping[str, str] = MappingProxyType({}),
    *,
    master_key_environment: str | None = GATEWAY_KEY,
) -> Iterator[NativeGateway]:
    """Boot the standalone binary from `config` and stop it with SIGTERM on exit, requiring status 0.

    `master_key_environment` is what `LITELLM_MASTER_KEY` holds in the gateway's environment. The
    binary itself never reads that variable; a config reaches it only through
    `master_key: os.environ/LITELLM_MASTER_KEY`.
    """
    binary: Final = gateway_binary()
    port: Final = allocate_port()
    config_path: Final = write_config(directory, config)
    log: Final = directory / "gateway.log"
    with (
        log.open("wb") as output,
        subprocess.Popen(
            (str(binary),),
            env=_environment(config_path, port, environment, master_key_environment),
            stdout=output,
            stderr=output,
            cwd=directory,
        ) as process,
    ):
        gateway: Final = NativeGateway(port, process, log)
        try:
            assert eventually(lambda: process.poll() is not None or _ready(gateway.url), bool, seconds=BOOT_SECONDS), (
                log.read_text()
            )
            assert process.poll() is None, log.read_text()
            yield gateway
        finally:
            status: Final = terminate(gateway)
            if status is None:
                kill_stray_process(process.pid)
                process.wait()
        assert status == 0, log.read_text()


@dataclass(frozen=True, slots=True)
class Refusal:
    returncode: int
    output: str


def refused_boot(
    config: Mapping[str, JsonValue] | None,
    directory: Path,
    environment: Mapping[str, str] = MappingProxyType({}),
    *,
    config_path: Path | None = None,
) -> Refusal:
    """Run the binary with a config that must stop the boot and return its exit status and output.

    A refused boot exits through `main`'s `Result`, so the output carries `Error: <Debug of the error>`
    (Rust's `Termination` for `Result<(), Box<dyn Error>>`)."""
    binary: Final = gateway_binary()
    path: Final = config_path if config_path is not None else write_config(directory, config or {})
    port: Final = allocate_port()
    completed: Final = subprocess.run(
        (str(binary),),
        env=_environment(path, port, environment, GATEWAY_KEY),
        capture_output=True,
        timeout=BOOT_SECONDS,
        cwd=directory,
        check=False,
    )
    return Refusal(completed.returncode, (completed.stdout + completed.stderr).decode(errors="replace"))


def marker() -> str:
    return f"dec1-gwnative-{uuid.uuid4().hex[:12]}"


def json_of(request: Request) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(request.body)


def response_json(response: httpx.Response) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(response.content)


def sse_events(text: str) -> tuple[tuple[str, dict[str, JsonValue]], ...]:
    """`(event name, data)` for every SSE block that has a JSON `data:` line; `[DONE]` is skipped."""
    events: list[tuple[str, dict[str, JsonValue]]] = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        name = ""
        data = ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data = line.removeprefix("data:").strip()
        if data and data != "[DONE]":
            events.append((name, JSON_OBJECT.validate_json(data)))
    return tuple(events)


Responder = Callable[[Request], Reply]


@dataclass
class ScriptedProvider:
    """One owned HTTP peer that answers as every provider, scripted per prompt marker."""

    wire: Wire
    scripts: dict[str, Responder] = field(default_factory=dict)
    seen: list[Request] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def url(self) -> str:
        return self.wire.url

    def script(self, case: str, responder: Responder) -> None:
        with self.lock:
            self.scripts[case] = responder

    def requests_for(self, case: str) -> tuple[Request, ...]:
        with self.lock:
            self.seen.extend(self.wire.drain())
            return tuple(request for request in self.seen if case.encode() in request.body)

    def requests_to(self, path_fragment: str) -> tuple[Request, ...]:
        with self.lock:
            self.seen.extend(self.wire.drain())
            return tuple(request for request in self.seen if path_fragment in request.target)


def _default_reply(request: Request) -> Reply:
    target: Final = request.target
    if target.endswith("/_oauth/token"):
        return Reply(
            body=json.dumps({"access_token": VERTEX_TOKEN, "expires_in": 3600, "token_type": "Bearer"}).encode()
        )
    if target.endswith("/converse"):
        return converse_reply("scripted converse reply")
    if "/v1/messages" in target or target.endswith("/invoke") or ":rawPredict" in target:
        return anthropic_reply("scripted anthropic reply")
    if target.endswith("/responses"):
        return responses_reply("scripted responses reply")
    if target.endswith("/chat/completions"):
        return openai_chat_reply("scripted chat reply")
    return Reply(status=404, body=b'{"error": {"message": "scripted 404"}}')


@contextmanager
def scripted_provider() -> Iterator[ScriptedProvider]:
    holder: list[ScriptedProvider] = []

    def respond(request: Request) -> Reply:
        provider: Final = holder[0]
        with provider.lock:
            scripted: Final = [
                responder for case, responder in provider.scripts.items() if case.encode() in request.body
            ]
        return scripted[0](request) if scripted else _default_reply(request)

    with wire_server(respond, keep_alive=True) as wire:
        holder.append(ScriptedProvider(wire))
        yield holder[0]


def anthropic_message(
    text: str, *, model: str = "claude-dec1-upstream", identity: str = "msg_dec1_gwnative"
) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 3},
    }


def anthropic_reply(
    text: str, *, model: str = "claude-dec1-upstream", headers: Mapping[str, str] = MappingProxyType({})
) -> Reply:
    return Reply(body=json.dumps(anthropic_message(text, model=model)).encode(), headers=headers)


def anthropic_error(status: int, error_type: str, message: str) -> Reply:
    return Reply(
        status=status, body=json.dumps({"type": "error", "error": {"type": error_type, "message": message}}).encode()
    )


def _sse(event: str, payload: Mapping[str, JsonValue]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n".encode()


def anthropic_stream_chunks(text: str, *, model: str = "claude-dec1-upstream") -> tuple[bytes, ...]:
    start: Final = {**anthropic_message("", model=model), "content": [], "stop_reason": None}
    return (
        _sse("message_start", {"type": "message_start", "message": start}),
        _sse(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        _sse(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        ),
        _sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 3},
            },
        ),
        _sse("message_stop", {"type": "message_stop"}),
    )


def anthropic_stream(
    text: str,
    *,
    abort_after: int | None = None,
    gate_after_first: threading.Event | None = None,
    pause: float = 0,
) -> Reply:
    return Reply(
        content_type="text/event-stream",
        chunks=anthropic_stream_chunks(text),
        abort_after=abort_after,
        gate_after_first=gate_after_first,
        pause_between_chunks=pause,
    )


def openai_chat_reply(text: str, *, model: str = "gpt-dec1-upstream") -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-dec1-gwnative",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            }
        ).encode()
    )


def openai_error(status: int, message: str, error_type: str = "server_error") -> Reply:
    return Reply(
        status=status, body=json.dumps({"error": {"message": message, "type": error_type, "code": None}}).encode()
    )


def converse_reply(text: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 5, "outputTokens": 3, "totalTokens": 8},
                "metrics": {"latencyMs": 1},
            }
        ).encode()
    )


def responses_object(
    text: str, *, identity: str = "resp_dec1_gwnative", model: str = "gpt-dec1-upstream"
) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": model,
        "output": [
            {
                "id": f"msg_{identity}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
    }


def responses_reply(text: str) -> Reply:
    return Reply(body=json.dumps(responses_object(text)).encode())


def responses_stream(text: str, *, abort_after: int | None = None) -> Reply:
    done: Final = responses_object(text)
    opened: Final = {**done, "status": "in_progress", "output": []}
    chunks: Final = (
        _sse("response.created", {"type": "response.created", "sequence_number": 0, "response": opened}),
        _sse(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "sequence_number": 1,
                "item_id": "msg_resp_dec1_gwnative",
                "output_index": 0,
                "content_index": 0,
                "delta": text,
            },
        ),
        _sse("response.completed", {"type": "response.completed", "sequence_number": 2, "response": done}),
    )
    return Reply(content_type="text/event-stream", chunks=chunks, abort_after=abort_after)


def deployment(name: str, model: str, **params: JsonValue) -> dict[str, JsonValue]:
    return {"model_name": name, "litellm_params": {"model": model, **params}}


def anthropic_deployment(name: str, api_base: str, **params: JsonValue) -> dict[str, JsonValue]:
    return deployment(name, "anthropic/claude-dec1-upstream", api_base=api_base, api_key=PROVIDER_KEY, **params)


def openai_like_deployment(name: str, api_base: str, **params: JsonValue) -> dict[str, JsonValue]:
    return deployment(name, "openai_like/gpt-dec1-upstream", api_base=f"{api_base}/v1", api_key=PROVIDER_KEY, **params)


def openai_deployment(name: str, api_base: str, **params: JsonValue) -> dict[str, JsonValue]:
    return deployment(name, "openai/gpt-dec1-upstream", api_base=f"{api_base}/v1", api_key=PROVIDER_KEY, **params)


def bedrock_deployment(
    name: str, api_base: str, model: str = "bedrock/anthropic.claude-dec1-v1"
) -> dict[str, JsonValue]:
    return deployment(
        name,
        model,
        api_base=api_base,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_region_name="us-east-1",
    )


def vertex_deployment(name: str, api_base: str, credentials: str) -> dict[str, JsonValue]:
    return deployment(
        name,
        "vertex_ai/claude-sonnet-4-5@20250929",
        api_base=api_base,
        vertex_project=VERTEX_PROJECT,
        vertex_location=VERTEX_LOCATION,
        vertex_credentials=credentials,
    )


def azure_ai_deployment(name: str, api_base: str) -> dict[str, JsonValue]:
    return deployment(name, "azure_ai/claude-dec1-upstream", api_base=f"{api_base}/", api_key=PROVIDER_KEY)


def azure_deployment(name: str, api_base: str) -> dict[str, JsonValue]:
    return deployment(
        name, "azure/gpt-dec1-deployment", api_base=api_base, api_key=PROVIDER_KEY, api_version="2024-10-21"
    )


def master_key_config(models: list[dict[str, JsonValue]], **sections: JsonValue) -> dict[str, JsonValue]:
    general: Final = {"master_key": "os.environ/LITELLM_MASTER_KEY"}
    extra_general = sections.pop("general_settings", {})
    assert isinstance(extra_general, dict)
    return {"model_list": models, "general_settings": {**general, **extra_general}, **sections}


def messages_body(case: str, model: str = "claude", **fields: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": case}], **fields}


def chat_body(case: str, model: str = "chat", **fields: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "messages": [{"role": "user", "content": case}], **fields}


def responses_body(case: str, model: str = "responses", **fields: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "input": case, **fields}


def openai_error_body(message: str, error_type: str, code: int) -> dict[str, JsonValue]:
    """The OpenAI envelope `gateway-inference/src/error.rs` `openai_body` writes."""
    return {"error": {"message": message, "type": error_type, "param": None, "code": code}}
