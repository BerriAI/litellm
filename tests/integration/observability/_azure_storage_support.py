import base64
import json
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final
from urllib.parse import parse_qs, unquote, urlsplit

import yaml
from integration._support.client import JsonValue, eventually, object_value
from integration._support.wire import Reply, Request

ACCOUNT: Final = "litellmaudit"
FILE_SYSTEM: Final = "litellm-logs"
SINK_HOSTS: Final = (f"{ACCOUNT}.dfs.core.localhost", f"{ACCOUNT}.blob.core.localhost")
ACCOUNT_KEY: Final = base64.b64encode(b"synthetic-account-key-for-integration-tests").decode()


@dataclass(slots=True)
class RecordingDataLakeSink:
    """Speaks enough of the Azure Data Lake Gen2 REST surface for the SDK's account-key upload: filesystem
    HEAD/PUT, blob HEAD for `exists`, PUT ?resource=directory|file, PATCH ?action=append|flush. Flushed
    files are kept by path and can be failed, delayed or served slowly for the chaos cells."""

    fail_status: int = 0
    delay_seconds: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)
    directories: set[str] = field(default_factory=set)  # mutable-ok: the sink is the durable store for the run
    pending: dict[str, bytearray] = field(default_factory=dict)  # mutable-ok: append lands before flush
    files: dict[str, bytes] = field(default_factory=dict)  # mutable-ok: flushed files must be readable later
    flush_count: dict[str, int] = field(default_factory=dict)  # mutable-ok: re-flush of one path means double upload
    in_flight: int = 0
    peak: int = 0
    attempt_count: int = 0

    def respond(self, request: Request) -> Reply:
        parts: Final = urlsplit(request.target)
        query: Final = {name: values[-1] for name, values in parse_qs(parts.query).items()}
        path: Final = unquote(parts.path)
        with self.lock:
            self.attempt_count += 1
            if self.fail_status:
                return Reply(status=self.fail_status, body=b'{"error":{"code":"SinkFailure"}}')
            if path != f"/{FILE_SYSTEM}" and not path.startswith(f"/{FILE_SYSTEM}/"):
                return Reply(status=400, body=b'{"error":{"code":"InvalidUri"}}')
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
        try:
            if self.delay_seconds:
                time.sleep(self.delay_seconds)
            with self.lock:
                return self._apply(request, path, query)
        finally:
            with self.lock:
                self.in_flight -= 1

    def _apply(self, request: Request, path: str, query: Mapping[str, str]) -> Reply:
        stamp: Final = {"etag": '"0x1"', "last-modified": "Thu, 01 Jan 2026 00:00:00 GMT", "x-ms-request-id": "sink"}
        empty: Final = "text/plain"
        if path == f"/{FILE_SYSTEM}":
            if request.method in ("HEAD", "GET"):
                return Reply(headers={**stamp, "x-ms-namespace-enabled": "true"}, body=b"{}", content_type=empty)
            if request.method == "PUT" and query.get("resource") == "filesystem":
                return Reply(status=201, headers=stamp, body=b"", content_type=empty)
            return Reply(status=400, body=b'{"error":{"code":"InvalidUri"}}')
        if request.method == "HEAD":
            if path in self.directories:
                return Reply(headers={**stamp, "x-ms-meta-hdi_isfolder": "true"}, body=b"", content_type=empty)
            if path in self.files:
                return Reply(headers=stamp, body=b"", content_type=empty)
            return Reply(status=404, headers={"x-ms-error-code": "PathNotFound"}, body=b"", content_type=empty)
        if request.method == "GET":
            if path in self.files:
                return Reply(headers=stamp, body=self.files[path])
            return Reply(status=404, headers={"x-ms-error-code": "PathNotFound"}, body=b"", content_type=empty)
        if request.method == "PUT":
            if query.get("resource") == "directory":
                self.directories.add(path)
                return Reply(status=201, headers=stamp, body=b"", content_type=empty)
            assert query.get("resource") == "file", request.target
            self.pending[path] = bytearray()
            return Reply(status=201, headers=stamp, body=b"", content_type=empty)
        assert request.method == "PATCH", request.method
        if query.get("action") == "append":
            assert int(query["position"]) == len(self.pending[path]), request.target
            self.pending[path].extend(request.body)
            return Reply(status=202, headers=stamp, body=b"", content_type=empty)
        assert query.get("action") == "flush", request.target
        assert int(query["position"]) == len(self.pending[path]), request.target
        self.files[path] = bytes(self.pending.pop(path))
        self.flush_count[path] = self.flush_count.get(path, 0) + 1
        return Reply(status=200, headers=stamp, body=b"", content_type=empty)

    def attempts(self) -> int:
        with self.lock:
            return self.attempt_count

    def duplicated(self) -> tuple[str, ...]:
        with self.lock:
            return tuple(path for path, count in self.flush_count.items() if count > 1)

    def stored(self) -> Mapping[str, bytes]:
        with self.lock:
            return MappingProxyType(dict(self.files))

    def payloads(self) -> Mapping[str, dict[str, JsonValue]]:
        return MappingProxyType({path: object_value(json.loads(body)) for path, body in self.stored().items()})


def azure_storage_config(
    path: Path, settings: Mapping[str, JsonValue] | None = None, *, callback_setting: str = "callbacks"
) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"].update({callback_setting: ["azure_storage"], **(settings or {})})
    target: Final = path / "azure_storage.yaml"
    target.write_text(yaml.safe_dump(config))
    return target


def azure_storage_environment(sink_url: str, cert_file: Path) -> Mapping[str, str]:
    port: Final = urlsplit(sink_url).port
    return MappingProxyType(
        {
            "AZURE_STORAGE_ACCOUNT_NAME": ACCOUNT,
            "AZURE_STORAGE_FILE_SYSTEM": FILE_SYSTEM,
            "AZURE_STORAGE_ACCOUNT_KEY": ACCOUNT_KEY,
            "AZURE_STORAGE_ENDPOINT_SUFFIX": f"core.localhost:{port}",
            "SSL_CERT_FILE": str(cert_file),
        }
    )


def collect_files(sink: RecordingDataLakeSink, count: int, seconds: float = 60) -> tuple[dict[str, JsonValue], ...]:
    """Wait until `count` flushed files exist, then return every stored payload."""
    eventually(lambda: len(sink.stored()), lambda total: total >= count, seconds=seconds)
    return tuple(sink.payloads().values())
