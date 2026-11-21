"""Shared rig for the `test_dec1_deploy_*` contracts: the deployment lifecycle of one owned proxy.

Every contract in these files starts its own proxy process (through `owned_proxy_process`, or
`refused_boot_log` for a boot that must fail) and then looks at what an operator or an orchestrator sees
from outside: the probes, the exit status, the log, the database rows a worker owns, and what survives a
restart. The scripted provider here answers OpenAI chat completions, unary and streamed, and can hold a
request in flight until the test releases it, so a test can send SIGTERM while work is still running.

Assumptions shared by every file:

* The harness runs the proxy as `python -m integration._support.proxy`, which maps the SIGTERM that uvicorn
  re-raises after its graceful shutdown to `sys.exit(0)` (tests/integration/_support/proxy.py:6). Exit code
  assertions are therefore about the proxy plus that wrapper, which is what a container entrypoint sees.
* A private scratch Postgres and Redis are booted for each owned proxy unless the test passes its own
  `DATABASE_URL` (tests/integration/_support/process.py:65), so database rows read through
  `owned.gateway.rows` belong to that proxy alone.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import yaml
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, eventually
from tests.integration._support.openai_wire import chat_reply
from tests.integration._support.ports import allocate_port
from tests.integration._support.process import OwnedProxy, proxy_database_environment, stop_child
from tests.integration._support.wire import Reply, Request, Wire, wire_server

CHAT: Final = "dec1-deploy-chat"
SECOND_CHAT: Final = "dec1-deploy-second"
PROVIDER_KEY: Final = "sk-dec1-deploy-provider"
REPLY_TEXT: Final = "dec1 deploy reply"
HOLD: Final = "dec1-deploy-hold"
HELD_SECONDS: Final = 120
INPUT_COST: Final = 0.001
OUTPUT_COST: Final = 0.002
EXPECTED_SPEND: Final = 5 * INPUT_COST + 3 * OUTPUT_COST
"""The provider reports 5 prompt and 3 completion tokens on every reply (tests/integration/_support/openai_wire.py:12)."""

READY: Final[Mapping[str, JsonValue]] = {"status": "healthy", "db": "connected"}
NOT_CONNECTED: Final[Mapping[str, JsonValue]] = {"status": "healthy", "db": "Not connected"}
SHUTTING_DOWN: Final[Mapping[str, JsonValue]] = {"status": "shutting_down"}
ALIVE: Final = "I'm alive!"
BUDGET_PREFIX: Final = "LiteLLM Proxy: Postgres connection budget per pod = "
BUDGET_SUFFIX: Final = (
    " connections; keep pods x that figure under max_connections minus superuser_reserved_connections"
)
SETUP_COMPLETE: Final = "LiteLLM: Setup complete. Skipping server startup as requested."
FALLBACK_SUFFIX: Final = ", so Python serves the public port without Rust forwarding"
NO_DATABASE: Final = ("DATABASE_URL", "DATABASE_URL_READ_REPLICA", "DIRECT_URL")
NO_REDIS: Final = ("REDIS_HOST", "REDIS_PORT", "REDIS_URL", "REDIS_PASSWORD")
_ANSI: Final = re.compile(r"\x1b\[[0-9;]*m")
_MODEL_DISCOVERY: Final = ("GET", "/v1/models")


@dataclass(frozen=True, slots=True)
class Provider:
    """A scripted OpenAI peer. A chat request whose last message contains `HOLD` sets `arrived` and is held
    until `release` is set: a unary call is answered only then, a streamed call sends its first chunk and
    waits there."""

    wire: Wire
    arrived: threading.Event
    release: threading.Event

    @property
    def api_base(self) -> str:
        return f"{self.wire.url}/v1"

    def posted(self) -> tuple[Request, ...]:
        return tuple(request for request in self.wire.drain() if (request.method, request.target) != _MODEL_DISCOVERY)


def _last_text(request: Request) -> str:
    body: Final = JSON_OBJECT.validate_json(request.body or b"{}")
    messages: Final = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    last: Final = messages[-1]
    return str(last.get("content", "")) if isinstance(last, dict) else ""


def _streamed(request: Request) -> bool:
    return JSON_OBJECT.validate_json(request.body or b"{}").get("stream") is True


def _responder(arrived: threading.Event, release: threading.Event) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if (request.method, request.target) == _MODEL_DISCOVERY:
            return Reply(body=b'{"object":"list","data":[]}')
        identity: Final = f"chatcmpl-{uuid.uuid4().hex}"
        stream: Final = _streamed(request)
        reply: Final = chat_reply(identity, "gpt-4o-mini", REPLY_TEXT, stream=stream)
        if HOLD not in _last_text(request):
            return reply
        arrived.set()
        if stream:
            return dataclasses.replace(reply, gate_after_first=release)
        release.wait(HELD_SECONDS)
        return reply

    return respond


@contextmanager
def provider() -> Iterator[Provider]:
    arrived: Final = threading.Event()
    release: Final = threading.Event()
    with wire_server(_responder(arrived, release)) as wire:
        try:
            yield Provider(wire, arrived, release)
        finally:
            release.set()


def deployment(name: str, api_base: str, **params: JsonValue) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {
            "model": "openai/gpt-4o-mini",
            "api_base": api_base,
            "api_key": PROVIDER_KEY,
            "input_cost_per_token": INPUT_COST,
            "output_cost_per_token": OUTPUT_COST,
            **params,
        },
    }


def deploy_config(
    api_base: str,
    *,
    general_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
    extra_models: tuple[dict[str, JsonValue], ...] = (),
) -> dict[str, JsonValue]:
    """One chat deployment on the scripted provider, the master key from the environment, no retries."""
    return {
        "model_list": [deployment(CHAT, api_base), *extra_models],
        "general_settings": {"master_key": "os.environ/LITELLM_MASTER_KEY", **(general_settings or {})},
        "litellm_settings": dict(litellm_settings or {}),
        "router_settings": {"num_retries": 0},
    }


def write_config(directory: Path, config: Mapping[str, object], name: str = "dec1-deploy.yaml") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path: Final = directory / name
    path.write_text(yaml.safe_dump(dict(config), sort_keys=False))
    return path


def marker() -> str:
    return f"dec1-deploy-{uuid.uuid4().hex}"


def digest(key: str) -> str:
    """How the proxy stores a virtual key: the sha256 hex of the raw key."""
    return sha256(key.encode()).hexdigest()


def chat(
    gateway: Gateway,
    *,
    text: str | None = None,
    model: str = CHAT,
    key: str | None = None,
    path: str = "/v1/chat/completions",
) -> httpx.Response:
    return gateway.request(
        "POST", path, {"model": model, "messages": [{"role": "user", "content": text or marker()}]}, key=key
    )


def stream_chat(gateway: Gateway, *, text: str | None = None, key: str | None = None) -> tuple[int, str]:
    body: Final[dict[str, JsonValue]] = {
        "model": CHAT,
        "stream": True,
        "messages": [{"role": "user", "content": text or marker()}],
    }
    with gateway.client.stream(
        "POST",
        "/v1/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {key or gateway.key}"},
        timeout=HELD_SECONDS + 30,
    ) as response:
        return response.status_code, response.read().decode()


def in_background(call: Callable[[], object]) -> Future[object]:
    """Run `call` on a fresh thread and return its future; the executor is not kept."""
    executor: Final = ThreadPoolExecutor(max_workers=1)
    future: Final = executor.submit(call)
    executor.shutdown(wait=False)
    return future


def held_chat(gateway: Gateway, upstream: Provider, *, stream: bool = False) -> Future[object]:
    """Start one chat call the provider holds, and return once the provider has it."""
    text: Final = f"{HOLD} {marker()}"
    future: Final = in_background(
        (lambda: stream_chat(gateway, text=text)) if stream else (lambda: chat(gateway, text=text))
    )
    assert upstream.arrived.wait(60), "the held request never reached the provider"
    return future


def terminate(owned: OwnedProxy, seconds: float = 90) -> int:
    """Send SIGTERM to the proxy root and return its exit status once it exits."""
    owned.begin_shutdown()
    return owned.process.wait(timeout=seconds)


def log_text(owned: OwnedProxy) -> str:
    return _ANSI.sub("", owned.log.read_text(errors="replace"))


def plain(log: str) -> str:
    return _ANSI.sub("", log)


def budget_line(log: str) -> str:
    """The single Postgres connection budget line the CLI prints before database setup."""
    lines: Final = [line.strip() for line in plain(log).splitlines() if BUDGET_PREFIX in line]
    assert len(lines) == 1, lines
    return lines[0][lines[0].index(BUDGET_PREFIX) :]


def readiness(gateway: Gateway, path: str = "/health/readiness") -> tuple[int, JsonValue]:
    response: Final = gateway.client.get(path)
    return response.status_code, response.json()


def heartbeat_rows(gateway: Gateway) -> list[dict[str, JsonValue]]:
    return gateway.rows('SELECT worker_id, hostname FROM "LiteLLM_ProxyWorkerHeartbeat" ORDER BY worker_id', ())


def await_rows(
    read: Callable[[], list[dict[str, JsonValue]]], count: int, seconds: float = 90
) -> list[dict[str, JsonValue]]:
    return eventually(read, lambda rows: len(rows) == count, seconds=seconds)


def response_id(response: httpx.Response) -> str:
    return str(JSON_OBJECT.validate_json(response.content)["id"])


def sse_frames(body: str) -> list[str]:
    return [line[len("data: ") :] for line in body.splitlines() if line.startswith("data: ")]


def streamed_text(body: str) -> str:
    pieces: Final[list[str]] = []
    for frame in sse_frames(body):
        if frame == "[DONE]":
            continue
        for choice in json.loads(frame).get("choices", []):
            pieces.append(choice.get("delta", {}).get("content") or "")
    return "".join(pieces)


@dataclass(frozen=True, slots=True)
class BootOutcome:
    """How a boot ended: `ready` when `/health/readiness` answered 200 before the deadline, otherwise the
    process exited first and `exit_code` holds its status."""

    ready: bool
    exit_code: int | None
    log: str


def _proxy_root() -> Path:
    return Path(os.environ.get("INTEGRATION_PROXY_ROOT") or Path(__file__).resolve().parents[3])


def _answers_ready(port: int) -> bool:
    try:
        with httpx.Client(timeout=2, trust_env=False) as client:
            return client.get(f"http://127.0.0.1:{port}/health/readiness").status_code == 200
    except httpx.TransportError:
        return False


def boot_outcome(
    gateway: Gateway,
    directory: Path,
    overrides: Mapping[str, str],
    *,
    config: Path,
    seconds: float = 300,
) -> BootOutcome:
    """Start the proxy the way `refused_boot_log` does (one worker, `--use_prisma_db_push`, the harness
    environment plus `overrides`) and report whether it became ready or exited, without asserting either.

    For contracts whose observable is a side effect (a table that must not exist, a log line) when the boot
    result itself is not the contract. Assumes the harness environment already puts `tests/` on the import
    path for `python -m integration._support.proxy`, as for every owned proxy.
    """
    directory.mkdir(parents=True, exist_ok=True)
    port: Final = allocate_port()
    log_path: Final = directory / f"dec1-deploy-boot-{uuid.uuid4().hex}.log"
    environment: Final = {
        **os.environ,
        **proxy_database_environment(),
        "LITELLM_MASTER_KEY": gateway.key,
        "STORE_MODEL_IN_DB": "True",
        "PYTHON_DOTENV_DISABLED": "1",
        **overrides,
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
            command, cwd=_proxy_root(), env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    try:
        deadline: Final = time.monotonic() + seconds
        while process.poll() is None and not _answers_ready(port):
            assert time.monotonic() < deadline, "the proxy neither exited nor became ready"
            time.sleep(0.2)
        ready: Final = process.poll() is None
        return BootOutcome(ready, None if ready else process.returncode, plain(log_path.read_text(errors="replace")))
    finally:
        stop_child(process, grace_seconds=30)
