from __future__ import annotations

import multiprocessing
from collections.abc import Generator
from contextlib import contextmanager
from multiprocessing.connection import Connection
from typing import ClassVar, Final

from ...shared.parity.local_server import LocalHttpHandler, LocalHttpServer
from .models import Backend

PYTHON_SENTINEL: Final = "litellm-benchmark-python"


class Provider(LocalHttpServer):
    def __init__(self, response: bytes, backend: Backend) -> None:
        super().__init__(("127.0.0.1", 0), Handler)
        self.response: Final = response
        self.backend: Final = backend


class Handler(LocalHttpHandler):
    disable_nagle_algorithm: ClassVar[bool] = True

    def do_POST(self) -> None:
        provider: Final = self.server
        assert isinstance(provider, Provider)
        self.rfile.read(int(self.headers.get("content-length", "0")))
        python_http: Final = self.headers.get("user-agent") == PYTHON_SENTINEL
        if python_http != (provider.backend == "python"):
            self.send_error(409, "SDK backend mismatch: Rust may have fallen back to Python")
            return
        if self.path != "/v1/ocr":
            self.send_error(404, "unexpected benchmark endpoint")
            return
        self.send_response_only(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(provider.response)))
        self.end_headers()
        self.wfile.write(provider.response)


def _serve(response: bytes, backend: Backend, pipe: Connection) -> None:
    with Provider(response, backend) as provider:
        pipe.send_bytes(provider.url.encode())
        pipe.close()
        provider.serve_forever()


@contextmanager
def provider_process(response: bytes, backend: Backend) -> Generator[str]:
    context: Final = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process: Final = context.Process(target=_serve, args=(response, backend, send))
    process.start()
    send.close()
    try:
        if not receive.poll(30):
            raise TimeoutError("benchmark provider did not start within 30 seconds")
        url: Final = receive.recv_bytes().decode()
        if not url.startswith("http://127.0.0.1:"):
            raise ValueError("benchmark provider returned an invalid local address")
        yield url
    finally:
        receive.close()
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)
        process.close()
