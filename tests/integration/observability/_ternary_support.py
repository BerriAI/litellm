from __future__ import annotations

import csv
import email.policy
import io
import socket
import threading
from collections.abc import Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import urlsplit

import yaml
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import TypeAdapter

_CONFIG_ADAPTER: Final = TypeAdapter(dict[str, object])


@dataclass(frozen=True, slots=True)
class CsvRow:
    values: Mapping[str, str]
    tags: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class MultipartRecord:
    path: str
    headers: Mapping[str, str]
    field_name: str
    filename: str
    content: bytes
    rows: tuple[CsvRow, ...]
    status: int


@dataclass(frozen=True, slots=True)
class BoundSink:
    url: str
    server: ThreadingHTTPServer
    thread: threading.Thread


_CSV_ROW_ADAPTER: Final = TypeAdapter(dict[str, str])
_TAGS_ADAPTER: Final = TypeAdapter(dict[str, str])


def parse_multipart(request: Request, status: int = 200) -> MultipartRecord:
    content_type: Final = request.headers["content-type"]
    message: Final = BytesParser(policy=email.policy.default).parsebytes(
        b"Content-Type: " + content_type.encode() + b"\r\nMIME-Version: 1.0\r\n\r\n" + request.body
    )
    parts: Final = tuple(
        part for part in message.iter_parts() if part.get_param("name", header="content-disposition") == "csv"
    )
    assert len(parts) == 1, f"multipart request contained {len(parts)} csv fields"
    part: Final = parts[0]
    content: Final = part.get_payload(decode=True)
    assert content is not None, "csv multipart field had no content"
    filename: Final = part.get_filename()
    assert filename is not None, "csv multipart field had no filename"
    field_name: Final = part.get_param("name", header="content-disposition")
    assert field_name is not None, "csv multipart field had no field name"
    parsed_rows: Final = tuple(
        CsvRow(
            values=MappingProxyType(_CSV_ROW_ADAPTER.validate_python(row)),
            tags=MappingProxyType(_TAGS_ADAPTER.validate_json(row["Tags"])),
        )
        for row in csv.DictReader(io.StringIO(content.decode("utf-8")))
    )
    return MultipartRecord(
        path=request.target,
        headers=MappingProxyType(dict(request.headers)),
        field_name=field_name,
        filename=filename,
        content=content,
        rows=parsed_rows,
        status=status,
    )


class MultipartSink:
    def __init__(self, *, delay_seconds: float = 0) -> None:
        self._lock: Final = threading.Lock()
        self._records: tuple[MultipartRecord, ...] = ()
        self._server_context: AbstractContextManager[Wire] | None = None
        self._wire: Wire | None = None
        self.port: int | None = None
        self._in_flight = 0
        self._peak = 0
        self._status_override: int | None = None
        self._redirect_location: str | None = None
        self._fail_chunk_index: int | None = None
        self._fail_chunk_api_key_alias: str | None = None
        self._failed_chunk_once = False
        self._delay_seconds = delay_seconds

    @property
    def url(self) -> str:
        assert self.port is not None, "multipart sink has not started"
        return f"http://127.0.0.1:{self.port}"

    @property
    def records(self) -> tuple[MultipartRecord, ...]:
        with self._lock:
            return self._records

    @property
    def peak_in_flight(self) -> int:
        with self._lock:
            return self._peak

    @property
    def in_flight(self) -> int:
        with self._lock:
            return self._in_flight

    @property
    def wire(self) -> Wire:
        assert self._wire is not None, "multipart sink is not running"
        return self._wire

    def start(self) -> None:
        assert self._server_context is None, "multipart sink is already running"
        server_context: Final = wire_server(self.respond, port=self.port or 0)
        wire: Final = server_context.__enter__()
        self._server_context = server_context
        self._wire = wire
        self.port = int(urlsplit(wire.url).port or 0)

    def stop(self) -> None:
        assert self._server_context is not None, "multipart sink is not running"
        server_context: Final = self._server_context
        server_context.__exit__(None, None, None)
        self._server_context = None
        self._wire = None

    def set_status_override(self, status: int | None) -> None:
        with self._lock:
            self._status_override = status

    def set_redirect(self, location: str | None) -> None:
        with self._lock:
            self._redirect_location = location

    def set_fail_chunk_once(self, chunk_index: int | None, *, api_key_alias: str | None = None) -> None:
        with self._lock:
            self._fail_chunk_index = chunk_index
            self._fail_chunk_api_key_alias = api_key_alias
            self._failed_chunk_once = False

    def set_delay(self, seconds: float) -> None:
        with self._lock:
            self._delay_seconds = seconds

    def reset_peak_in_flight(self) -> None:
        with self._lock:
            self._peak = 0

    def respond(self, request: Request) -> Reply:
        with self._lock:
            self._in_flight += 1
            self._peak = max(self._peak, self._in_flight)
        try:
            record: Final = parse_multipart(request)
            chunk_index: Final = int(record.headers.get("x-ternary-chunk-index", "0"))
            with self._lock:
                failure_alias_matches: Final = self._fail_chunk_api_key_alias is None or any(
                    row.tags.get("api_key_alias") == self._fail_chunk_api_key_alias for row in record.rows
                )
                failed_chunk: Final = (
                    self._fail_chunk_index == chunk_index and not self._failed_chunk_once and failure_alias_matches
                )
                redirect_location: Final = None if failed_chunk else self._redirect_location
                status: Final = (
                    500 if failed_chunk else self._status_override or (307 if redirect_location is not None else 200)
                )
                delay_seconds: Final = self._delay_seconds
                if failed_chunk:
                    self._failed_chunk_once = True
                self._records = (
                    *self._records,
                    MultipartRecord(
                        path=record.path,
                        headers=record.headers,
                        field_name=record.field_name,
                        filename=record.filename,
                        content=record.content,
                        rows=record.rows,
                        status=status,
                    ),
                )
            if delay_seconds:
                threading.Event().wait(delay_seconds)
            reply_headers: Final = (
                MappingProxyType({"Location": redirect_location})
                if redirect_location is not None
                else MappingProxyType({})
            )
            return Reply(status=status, body=b"recorded", content_type="text/plain", headers=reply_headers)
        finally:
            with self._lock:
                self._in_flight -= 1


@contextmanager
def running_sink(sink: MultipartSink) -> Generator[MultipartSink, None, None]:
    sink.start()
    try:
        yield sink
    finally:
        if sink._server_context is not None:
            sink.stop()


@contextmanager
def bound_sink_server(sink: MultipartSink, host: str) -> Generator[BoundSink, None, None]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            request: Final = Request(
                self.command,
                self.path,
                MappingProxyType({name.lower(): value for name, value in self.headers.items()}),
                self.rfile.read(int(self.headers.get("content-length", "0"))),
            )
            reply: Final = sink.respond(request)
            self.send_response(reply.status)
            self.send_header("content-type", reply.content_type)
            self.send_header("content-length", str(len(reply.body)))
            self.send_header("connection", "close")
            for name, value in reply.headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(reply.body)
            self.close_connection = True

        def log_message(self, format: str, *args: object) -> None:
            return

    server: Final = ThreadingHTTPServer((host, 0), Handler)
    thread: Final = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    thread.start()
    try:
        yield BoundSink(f"http://{host}:{server.server_port}", server, thread)
    finally:
        server.shutdown()
        thread.join(timeout=6)
        server.server_close()
        assert not thread.is_alive(), "bound multipart sink survived cleanup"


def non_loopback_ipv4() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
        connection.connect(("192.0.2.1", 80))
        address: Final = str(connection.getsockname()[0])
    assert not address.startswith("127."), f"selected loopback address {address}"
    return address


def write_proxy_config(directory: Path, callbacks: tuple[str, ...], *, redis_db: int | None = None) -> Path:
    base: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
    config: Final = _CONFIG_ADAPTER.validate_python(yaml.safe_load(base.read_text()))
    settings: Final = _CONFIG_ADAPTER.validate_python(config["litellm_settings"])
    cache_params: Final = _CONFIG_ADAPTER.validate_python(settings["cache_params"])
    additional_cache_params: Final = {} if redis_db is None else {"db": redis_db}
    configured_settings: Final = {
        **settings,
        "callbacks": list(callbacks),
        "cache_params": {**cache_params, **additional_cache_params},
    }
    configured: Final = {**config, "litellm_settings": configured_settings}
    destination: Final = directory / "proxy_config.yaml"
    destination.write_text(yaml.safe_dump(configured, sort_keys=False))
    return destination


def ternary_environment(
    sink_url: str,
    *,
    interval_seconds: str = "1",
    excluded: frozenset[str] = frozenset(),
) -> Mapping[str, str]:
    values: Final = {
        "TERNARY_API_KEY": "synthetic-ternary-api-key",
        "TERNARY_CONNECTION_ID": "conn:audit@1",
        "TERNARY_BASE_URL": sink_url,
        "TERNARY_EXPORT_FREQUENCY": "interval",
        "TERNARY_EXPORT_INTERVAL_SECONDS": interval_seconds,
    }
    return MappingProxyType({name: value for name, value in values.items() if name not in excluded})


def vantage_environment(sink_url: str) -> Mapping[str, str]:
    return MappingProxyType(
        {
            "VANTAGE_API_KEY": "synthetic-vantage-api-key",
            "VANTAGE_INTEGRATION_TOKEN": "synthetic-vantage-token",
            "VANTAGE_BASE_URL": sink_url,
            "VANTAGE_EXPORT_FREQUENCY": "interval",
            "VANTAGE_EXPORT_INTERVAL_SECONDS": "1",
        }
    )


def uploads_for_alias(records: tuple[MultipartRecord, ...], alias: str) -> tuple[tuple[MultipartRecord, CsvRow], ...]:
    return tuple(
        chain.from_iterable(
            tuple((record, row) for row in record.rows if row.tags.get("api_key_alias") == alias) for record in records
        )
    )


def row_by_alias(record: MultipartRecord, alias: str) -> tuple[CsvRow, ...]:
    return tuple(row for row in record.rows if row.tags.get("api_key_alias") == alias)
