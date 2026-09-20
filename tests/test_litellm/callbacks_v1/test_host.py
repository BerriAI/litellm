"""The extension host worker, driven over a socketpair the way a gateway would drive it."""

import asyncio
import os
import socket
import subprocess
import sys
import threading
from collections.abc import Generator
from pathlib import Path
from types import MappingProxyType

import pytest

from litellm import callbacks_v1
from litellm.callbacks_v1 import EnvelopeV1, RequestFactsV1, WirePatchV1
from litellm.callbacks_v1.host import INTERCEPT_THREADS, ProtocolError, Worker, encode_frame, read_frame
from tests.test_litellm.callbacks_v1.builtin.support import golden


@pytest.fixture(autouse=True)
def clean_registry() -> Generator[None]:
    yield
    for subscriber in callbacks_v1.snapshot():
        callbacks_v1.unregister(subscriber)


class Gateway:
    """The gateway's end of the socket: starts a worker on a thread and speaks frames to it."""

    def __init__(self) -> None:
        self.sock, worker_sock = socket.socketpair()
        self.sock.settimeout(5)
        self.thread = threading.Thread(target=Worker(worker_sock, callbacks_v1.snapshot()).serve, daemon=True)
        self.thread.start()
        self.hello = self.read()

    def send(self, **frame: object) -> None:
        self.sock.sendall(encode_frame(frame))

    def read(self) -> dict:
        frame = read_frame(self.sock)
        assert frame is not None
        return dict(frame)

    def flush(self) -> list[dict]:
        """Everything the worker said before it confirmed the flush."""
        self.send(type="flush", id="f")
        frames = []
        while (frame := self.read()) != {"type": "flushed", "id": "f"}:
            frames.append(frame)
        return frames

    def close(self) -> None:
        self.sock.close()
        self.thread.join(5)
        assert not self.thread.is_alive()


def test_hello_reports_every_subscription_as_data() -> None:
    @callbacks_v1.on_event("call.succeeded", "call.failed", name="recorder")
    def record(event: EnvelopeV1) -> None: ...

    @callbacks_v1.before_send
    async def patch(request: RequestFactsV1) -> WirePatchV1 | None: ...

    gateway = Gateway()

    assert gateway.hello == {
        "type": "hello",
        "schema": 1,
        "subscriptions": [
            {"name": "recorder", "events": ["call.failed", "call.succeeded"], "observes": True, "intercepts": False},
            {"name": f"{__name__}.{patch.__qualname__}", "events": [], "observes": False, "intercepts": True},
        ],
    }
    gateway.close()


def test_one_subscriber_gets_its_envelopes_in_the_order_they_were_sent() -> None:
    seen: list[int] = []

    @callbacks_v1.on_event("response.received", name="ordered")
    async def record(event: EnvelopeV1) -> None:
        seen.append(event["seq"])

    gateway = Gateway()
    for seq in range(50):
        gateway.send(type="event", subscriber=0, envelope={**golden("response.received"), "seq": seq})

    assert gateway.flush() == []
    assert seen == list(range(50))
    gateway.close()


def test_an_observer_that_raises_is_reported_and_the_next_envelope_still_arrives() -> None:
    seen: list[str] = []

    @callbacks_v1.on_event("call.started", "call.failed", name="flaky")
    def record(event: EnvelopeV1) -> None:
        seen.append(event["event"]["type"])
        if event["event"]["type"] == "call.started":
            raise ValueError("boom")

    gateway = Gateway()
    gateway.send(type="event", subscriber=0, envelope=golden("call.started"))
    gateway.send(type="event", subscriber=0, envelope=golden("call.failed"))

    assert gateway.flush() == [{"type": "report", "subscriber": 0, "event": "call.started", "message": "boom"}]
    assert seen == ["call.started", "call.failed"]
    gateway.close()


def test_an_interceptor_answers_with_its_patch_even_from_a_read_only_view() -> None:
    @callbacks_v1.before_send
    def patch(request: RequestFactsV1) -> WirePatchV1 | None:
        return {"body": MappingProxyType({"model": request["model"], "blocks": ({"a": 1},)})}

    gateway = Gateway()
    gateway.send(type="intercept", id=7, subscriber=0, request=golden("request.sending")["event"])

    assert gateway.read() == {"type": "patch", "id": 7, "patch": {"body": {"model": "model", "blocks": [{"a": 1}]}}}
    gateway.close()


def test_an_interceptor_that_raises_is_an_error_for_the_gateway_to_act_on() -> None:
    @callbacks_v1.before_send
    async def patch(request: RequestFactsV1) -> WirePatchV1 | None:
        raise PermissionError("not this model")

    gateway = Gateway()
    gateway.send(type="intercept", id="a", subscriber=0, request=golden("request.sending")["event"])

    assert gateway.read() == {
        "type": "error",
        "id": "a",
        "error_class": "builtins.PermissionError",
        "message": "not this model",
    }
    gateway.close()


def test_an_interception_does_not_wait_behind_a_slow_observer() -> None:
    release = threading.Event()

    @callbacks_v1.on_event("call.started", name="slow")
    def slow(event: EnvelopeV1) -> None:
        release.wait(5)

    @callbacks_v1.before_send
    def patch(request: RequestFactsV1) -> WirePatchV1 | None:
        return None

    gateway = Gateway()
    gateway.send(type="event", subscriber=0, envelope=golden("call.started"))
    gateway.send(type="intercept", id=1, subscriber=1, request=golden("request.sending")["event"])

    assert gateway.read() == {"type": "patch", "id": 1, "patch": None}
    release.set()
    gateway.close()


def test_every_async_interception_is_in_flight_at_once() -> None:
    """Concurrency is the loop's, not a pool size: an async handler that only waits holds no
    thread, so far more interceptions run together than there are intercept threads. Awaiting
    each other is how they prove it — none of them can finish until all of them have started."""
    burst = INTERCEPT_THREADS * 4
    started = 0
    everyone: asyncio.Event | None = None

    @callbacks_v1.before_send
    async def patch(request: RequestFactsV1) -> WirePatchV1 | None:
        nonlocal started, everyone
        if everyone is None:  # every handler runs on the one loop, so this races with nothing
            everyone = asyncio.Event()
        started += 1
        if started == burst:
            everyone.set()
        await asyncio.wait_for(everyone.wait(), 3)
        return None

    gateway = Gateway()
    for frame_id in range(burst):
        gateway.send(type="intercept", id=frame_id, subscriber=0, request=golden("request.sending")["event"])

    answers = sorted((gateway.read() for _ in range(burst)), key=lambda frame: frame["id"])
    assert answers == [{"type": "patch", "id": frame_id, "patch": None} for frame_id in range(burst)]
    gateway.close()


def test_a_frame_the_protocol_does_not_know_stops_the_worker() -> None:
    sock, worker_sock = socket.socketpair()
    worker = Worker(worker_sock, ())
    sock.sendall(encode_frame({"type": "shell", "command": "id"}))

    with pytest.raises(ProtocolError, match="unknown frame type"):
        worker.serve()
    sock.close()


CALLBACK_MODULE = """
import os
from litellm.callbacks_v1 import before_send

@before_send
def leak(request):
    return {"body": {"env": sorted(os.environ), "cwd_files": sorted(os.listdir("."))}}
"""


def test_a_spawned_worker_sees_only_the_environment_it_was_given(tmp_path: Path) -> None:
    modules = tmp_path / "modules"
    modules.mkdir()
    (modules / "leaky.py").write_text(CALLBACK_MODULE)
    empty = tmp_path / "empty"
    empty.mkdir()
    gateway_sock, worker_sock = socket.socketpair()
    gateway_sock.settimeout(60)
    repo = Path(__file__).parents[3]
    given = {"PYTHONPATH": os.pathsep.join((str(repo), str(modules))), "LITELLM_MODE": "PRODUCTION"}

    process = subprocess.Popen(
        [sys.executable, "-m", "litellm.callbacks_v1.host", "--fd", str(worker_sock.fileno()), "--load", "leaky"],
        pass_fds=(worker_sock.fileno(),),
        env=given,
        cwd=empty,
    )
    worker_sock.close()
    try:
        hello = read_frame(gateway_sock)
        assert hello is not None and [sub["name"] for sub in hello["subscriptions"]] == ["leaky.leak"]  # pyright: ignore[reportIndexIssue, reportGeneralTypeIssues]  # the hello frame's subscriptions are a list of objects
        gateway_sock.sendall(encode_frame({"type": "intercept", "id": 1, "subscriber": 0, "request": {}}))
        answer = read_frame(gateway_sock)
    finally:
        gateway_sock.close()
        assert process.wait(30) == 0

    assert answer is not None
    leaked = answer["patch"]["body"]  # pyright: ignore[reportIndexIssue, reportOptionalSubscript]  # the patch frame carries the callback's body
    assert set(leaked["env"]) - {"LC_CTYPE", "__CF_USER_TEXT_ENCODING"} <= set(given)
    assert leaked["cwd_files"] == []


GOLDEN_FRAMES = Path(__file__).parents[3] / "litellm-rust/crates/callbacks-v1-exthost/golden"


def golden_frame(name: str) -> dict:
    import json

    return json.loads((GOLDEN_FRAMES / name).read_text())


def test_the_worker_answers_the_golden_frames_rust_sends_with_the_golden_frames_rust_parses() -> None:
    """Both sides are pinned to one directory: Rust serialises `to_worker` and parses
    `from_worker`; here the worker is fed the first and must produce the shape of the second."""
    seen: list[object] = []

    @callbacks_v1.on_event("call.started", name="recorder")
    def record(event: EnvelopeV1) -> None:
        seen.append(event)
        raise ValueError("boom")

    @callbacks_v1.before_send
    def patch(request: RequestFactsV1) -> WirePatchV1 | None:
        seen.append(request)
        return {"headers": {"remove": ["X-Trace"], "set": [("X-Team", "core")]}, "body": {"input": "patched"}}

    event, intercept, flush = (golden_frame(f"to_worker/{name}.json") for name in ("event", "intercept", "flush"))
    gateway = Gateway()
    assert set(gateway.hello) == set(golden_frame("from_worker/hello.json"))
    assert {key for sub in gateway.hello["subscriptions"] for key in sub} == {
        key for sub in golden_frame("from_worker/hello.json")["subscriptions"] for key in sub
    }

    gateway.send(**intercept)
    assert gateway.read() == golden_frame("from_worker/patch.json")
    gateway.send(**event)
    gateway.send(**flush)
    assert gateway.read() == golden_frame("from_worker/report.json")
    assert gateway.read() == golden_frame("from_worker/flushed.json")
    assert seen == [intercept["request"], event["envelope"]]
    gateway.close()


def test_the_authoring_surface_and_the_worker_import_nothing_but_the_standard_library() -> None:
    """A worker that could run without the rest of litellm stays possible only while these two
    files need nothing from it. `typing_extensions` is the one third-party import."""
    import ast

    allowed = set(sys.stdlib_module_names) | {"typing_extensions", "litellm.callbacks_v1"}
    for path in ("litellm/callbacks_v1/__init__.py", "litellm/callbacks_v1/host.py"):
        tree = ast.parse((Path(__file__).parents[3] / path).read_text())
        imported = {
            alias.name if isinstance(node, ast.Import) else str(node.module)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert {name for name in imported if name not in allowed and name.split(".")[0] not in allowed} == set(), path
