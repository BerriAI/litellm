import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final

import pytest

from litellm.integrations.custom_logger_v2.call import CallContext
from litellm.integrations.custom_logger_v2.derivations import DERIVED_ATTEMPT_FIELDS
from litellm.integrations.custom_logger_v2.enums import Hook
from litellm.integrations.custom_logger_v2.enforcement import FieldView
from litellm.integrations.custom_logger_v2.field_sources import HookArgs, with_sources
from litellm.integrations.custom_logger_v2.legacy import KwargsFields
from litellm.integrations.custom_logger_v2.rust import WireManifest
from litellm.integrations.s3_v3 import S3Logger
from litellm.litellm_core_utils.litellm_logging import LOGGING_KEYS, V2_ASYNC_LOGGING_LINE
from litellm.types.integrations.custom_logger_v2 import Outcome

DROP: Final = -1


@dataclass(frozen=True)
class Received:
    method: str
    path: str
    headers: Mapping[str, str]
    body: bytes
    at: float


@dataclass
class ScriptedBucket:
    port: int = 0
    received: list[Received] = field(default_factory=list)  # mutable-ok: appended by the server thread
    objects: dict[str, bytes] = field(default_factory=dict)  # mutable-ok: GET serves what PUT stored
    script: list[tuple[str, list[tuple[int, str]]]] = field(default_factory=list)  # mutable-ok: replies are consumed
    delay: float = 0.0
    in_flight: int = 0
    peak: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def reply(self, fragment: str, *replies: tuple[int, str]) -> None:
        self.script.append((fragment, list(replies)))

    def puts(self) -> list[Received]:
        return [request for request in self.received if request.method == "PUT"]

    def next_reply(self, path: str) -> tuple[int, str]:
        with self.lock:
            for fragment, replies in self.script:
                if fragment in path and replies:
                    return replies.pop(0) if len(replies) > 1 else replies[0]
        return 200, ""


def _handler(bucket: ScriptedBucket) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_: object) -> None:
            return

        def _send(self, status: int, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_PUT(self) -> None:
            body: Final = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            headers: Final = {name.lower(): value for name, value in self.headers.items()}
            with bucket.lock:
                bucket.received.append(Received("PUT", self.path, headers, body, time.monotonic()))
                bucket.in_flight += 1
                bucket.peak = max(bucket.peak, bucket.in_flight)
            if bucket.delay:
                time.sleep(bucket.delay)
            with bucket.lock:
                bucket.in_flight -= 1
            status, text = bucket.next_reply(self.path)
            if status == DROP:
                self.close_connection = True
                return
            if status == 200:
                with bucket.lock:
                    bucket.objects[self.path] = body
            self._send(status, text.encode())

        def do_GET(self) -> None:
            headers: Final = {name.lower(): value for name, value in self.headers.items()}
            with bucket.lock:
                bucket.received.append(Received("GET", self.path, headers, b"", time.monotonic()))
                stored: Final = bucket.objects.get(self.path)
            if stored is None:
                self._send(404, b"<Error><Code>NoSuchKey</Code></Error>")
                return
            self._send(200, stored)

    return Handler


class _BucketServer(ThreadingHTTPServer):
    request_queue_size = 256


@contextmanager
def scripted_bucket() -> Iterator[ScriptedBucket]:
    bucket: Final = ScriptedBucket()
    server: Final = _BucketServer(("127.0.0.1", 0), _handler(bucket))
    bucket.port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield bucket
    finally:
        server.shutdown()


def rust_s3_logger(monkeypatch: pytest.MonkeyPatch, **params: object) -> S3Logger:  # kwargs-ok: the S3Logger keywords
    return S3Logger(params) if params else S3Logger()


async def log_attempt(
    logger: S3Logger, kwargs: Mapping[str, object], start_time: datetime, outcome: Outcome = Outcome.success
) -> None:
    args: Final = HookArgs(request=kwargs, start_time=start_time, end_time=start_time, outcome=outcome)
    view: Final = FieldView(with_sources(KwargsFields(kwargs, LOGGING_KEYS), DERIVED_ATTEMPT_FIELDS, args))
    manifest: Final = WireManifest.model_validate_json(logger.rust.manifest())
    reads: Final = next(s.reads for s in manifest.subscriptions if s.hook is Hook.attempt_ended)
    line: Final = V2_ASYNC_LOGGING_LINE.set(True)
    try:
        _ = await logger.run_rust(Hook.attempt_ended, reads, view, CallContext(Hook.attempt_ended, None, None))
    finally:
        V2_ASYNC_LOGGING_LINE.reset(line)


async def log_payload(logger: S3Logger, payload: Mapping[str, object], start_time: datetime) -> None:
    await log_attempt(logger, {"standard_logging_object": payload}, start_time)
