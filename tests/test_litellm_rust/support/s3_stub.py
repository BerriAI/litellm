"""In-process path-style S3 stub for native cache parity tests."""

import threading
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final
from urllib.parse import unquote, urlsplit

_STORED_HEADERS: Final = (
    "cache-control",
    "content-type",
    "content-language",
    "content-disposition",
    "expires",
)


@dataclass
class S3Object:
    body: bytes
    headers: dict[str, str] = field(default_factory=dict)


class S3Stub:
    """Minimal path-style S3 endpoint serving PUT and GET object operations."""

    def __init__(self) -> None:
        self._objects: dict[str, S3Object] = {}
        stub: Final = self

        class Handler(BaseHTTPRequestHandler):
            def _key(self) -> str:
                parts: Final = urlsplit(self.path).path.lstrip("/").split("/", 1)
                return unquote(parts[1]) if len(parts) == 2 else ""

            def _read_body(self) -> bytes:
                transfer: Final = self.headers.get("transfer-encoding", "")
                if "chunked" not in transfer:
                    return self.rfile.read(int(self.headers.get("content-length", 0)))
                chunks: Final = bytearray()
                while True:
                    size = int(self.rfile.readline().split(b";")[0].strip(), 16)
                    if size == 0:
                        while self.rfile.readline().strip():
                            pass
                        return bytes(chunks)
                    chunks.extend(self.rfile.read(size))
                    self.rfile.readline()

            def do_PUT(self) -> None:
                body: Final = self._read_body()
                headers: Final = {name: self.headers[name] for name in _STORED_HEADERS if name in self.headers}
                stub._objects = {**stub._objects, self._key(): S3Object(body=body, headers=headers)}
                self.send_response(200)
                self.send_header("ETag", '"stub"')
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_HEAD(self) -> None:
                self._object(send_body=False)

            def do_GET(self) -> None:
                self._object(send_body=True)

            def _object(self, send_body: bool) -> None:
                entry: Final = stub._objects.get(self._key())
                if entry is None:
                    self.send_response(404)
                    self.send_header("Content-Type", "application/xml")
                    body: Final = b'<?xml version="1.0" encoding="UTF-8"?><Error><Code>NoSuchKey</Code></Error>'
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    if send_body:
                        self.wfile.write(body)
                    return
                self.send_response(200)
                for name, value in entry.headers.items():
                    self.send_header(name, value)
                self.send_header("ETag", '"stub"')
                self.send_header("Content-Length", str(len(entry.body)))
                self.end_headers()
                if send_body:
                    self.wfile.write(entry.body)

            def log_message(self, format: str, *args: object) -> None:
                pass

        self._server: Final = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._worker: Final = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._worker.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    @property
    def objects(self) -> dict[str, S3Object]:
        return self._objects

    def put_object(self, key: str, body: bytes, headers: dict[str, str] | None = None) -> None:
        self._objects = {**self._objects, key: S3Object(body=body, headers=headers or {})}

    def expires(self, key: str) -> object:
        header: Final = self._objects[key].headers.get("expires")
        return parsedate_to_datetime(header) if header else None

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._worker.join(timeout=5)
