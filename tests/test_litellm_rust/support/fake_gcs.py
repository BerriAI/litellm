from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socket import socket
from types import MappingProxyType
from typing import Final, cast
from urllib.parse import unquote, urlsplit


@dataclass(frozen=True, slots=True)
class RecordedRequest:
    method: str
    path: str
    query: str
    headers: Mapping[str, str]
    body: bytes


class _FakeGcsHandler(BaseHTTPRequestHandler):
    def __init__(
        self,
        request: socket | tuple[bytes, socket],
        client_address: tuple[str, int],
        server: ThreadingHTTPServer,
        *,
        fake: FakeGcs,
    ) -> None:
        self._fake: Final = fake
        super().__init__(request, client_address, server)

    def _handle(self) -> None:
        parsed: Final = urlsplit(self.path)
        content_length: Final = int(self.headers.get("Content-Length", "0"))
        body: Final = self.rfile.read(content_length) if content_length else b""
        headers: Final = MappingProxyType(
            {name.title(): value for name, value in self.headers.items()}
        )
        self._fake.record(
            RecordedRequest(
                method=self.command,
                path=parsed.path,
                query=parsed.query,
                headers=headers,
                body=body,
            )
        )
        if self.headers.get("Authorization") != f"Bearer {self._fake.token}":
            self._send_json(401, {"error": "unauthorized"})
            return

        upload_prefix: Final = "/upload/storage/v1/b/"
        download_prefix: Final = "/storage/v1/b/"
        if parsed.path.startswith(upload_prefix) and parsed.path.endswith("/o"):
            self._upload(parsed.path[len(upload_prefix) : -2], parsed.query, body)
            return
        if parsed.path.startswith(download_prefix):
            self._download(parsed.path[len(download_prefix) :], parsed.query)
            return
        self._send_json(404, {"error": "not found"})

    def _upload(self, path: str, query: str, body: bytes) -> None:
        values: Final = {
            unquote(pair.partition("=")[0]): unquote(pair.partition("=")[2])
            for pair in query.split("&")
            if pair
        }
        if not path or values.get("uploadType") != "media" or "name" not in values:
            self._send_json(404, {"error": "not found"})
            return
        self._fake.put_object(path, values["name"], body)
        self._send_json(200, {"name": values["name"], "bucket": path})

    def _download(self, path: str, query: str) -> None:
        bucket, separator, encoded_name = path.partition("/o/")
        if not separator or query != "alt=media":
            self._send_json(404, {"error": "not found"})
            return
        name: Final = unquote(encoded_name)
        if name.endswith("/server-error") or name == "server-error":
            self._send_json(500, {"error": "server error"})
            return
        body: Final = self._fake.get_object(bucket, name)
        if body is None:
            self._send_json(404, {"error": "not found"})
            return
        self._send(200, body, "application/octet-stream")

    def _send_json(self, status: int, value: object) -> None:
        payload: Final = json.dumps(value).encode()
        self._send(status, payload, "application/json")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass

    do_GET = _handle
    do_POST = _handle


class FakeGcs:
    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], bytes] = {}  # mutable-ok: fake object store
        self._requests: list[RecordedRequest] = []  # mutable-ok: recorded request history
        self._server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            partial(_FakeGcsHandler, fake=self),
        )
        self._worker = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._worker.start()
        self.token: Final = "test-token"

    @property
    def url(self) -> str:
        address: Final = cast(tuple[str, int], self._server.server_address)
        host, port = address
        return f"http://{host}:{port}"

    @property
    def objects(self) -> Mapping[tuple[str, str], bytes]:
        return MappingProxyType(self._objects)

    @property
    def requests(self) -> tuple[RecordedRequest, ...]:
        return tuple(self._requests)

    def put(self, bucket: str, name: str, body: bytes) -> None:
        self.put_object(bucket, name, body)

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._worker.join(timeout=5)

    def record(self, request: RecordedRequest) -> None:
        self._requests.append(request)

    def put_object(self, bucket: str, name: str, body: bytes) -> None:
        self._objects[(bucket, name)] = body

    def get_object(self, bucket: str, name: str) -> bytes | None:
        return self._objects.get((bucket, name))
