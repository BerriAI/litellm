from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, TypeAlias
from urllib.parse import parse_qsl, urlsplit

from integration._support import tls
from integration._support.sigv4 import canonical_query, signature
from integration._support.wire import Reply
from pydantic import JsonValue

_AUTHORIZATION: Final = re.compile(
    r"^AWS4-HMAC-SHA256 Credential=(?P<key>[^/]+)/(?P<scope>[^,]+), "
    r"SignedHeaders=(?P<signed>[^,]+), Signature=(?P<signature>[0-9a-f]{64})$"
)
_BEARER: Final = re.compile(r"^Bearer (?P<token>.+)$")
UNKNOWN_CREDENTIAL: Final = "none"
NO_PROXY_HOSTS: Final = "127.0.0.1,localhost"
INFERENCE_PROFILES: Final = "/inference-profiles"
FOUNDATION_MODELS: Final = "/foundation-models"


@dataclass(frozen=True, slots=True)
class ControlPlaneRequest:
    host: str
    method: str
    target: str
    headers: Mapping[str, str]
    body: bytes

    @property
    def path(self) -> str:
        return urlsplit(self.target).path

    @property
    def query(self) -> Mapping[str, str]:
        return MappingProxyType(dict(parse_qsl(urlsplit(self.target).query, keep_blank_values=True)))

    @property
    def credential(self) -> str:
        """The access key id of a SigV4 request, the token of a bearer one, UNKNOWN_CREDENTIAL otherwise."""
        authorization: Final = self.headers.get("authorization", "")
        signed: Final = _AUTHORIZATION.match(authorization)
        if signed is not None:
            return signed.group("key")
        bearer: Final = _BEARER.match(authorization)
        return bearer.group("token") if bearer is not None else UNKNOWN_CREDENTIAL


def verified_scope(request: ControlPlaneRequest, secret: str) -> str:
    """Recompute the SigV4 signature with `secret`; returns the credential scope the request was signed under."""
    found: Final = _AUTHORIZATION.match(request.headers.get("authorization", ""))
    assert found is not None, dict(request.headers)
    parts: Final = urlsplit(request.target)
    _, expected = signature(
        request.method,
        parts.path,
        request.headers,
        found.group("signed"),
        request.body,
        secret,
        found.group("scope"),
        canonical_query(parts.query),
    )
    assert found.group("signature") == expected, request
    return found.group("scope")


Responder: TypeAlias = Callable[[ControlPlaneRequest], Reply]


def json_reply(status: int, payload: Mapping[str, JsonValue]) -> Reply:
    return Reply(status=status, body=json.dumps(payload).encode())


def _unregistered(request: ControlPlaneRequest) -> Reply:
    return json_reply(403, {"message": f"no scripted catalog for credential {request.credential}"})


@dataclass(frozen=True, slots=True)
class Catalog:
    """What the scripted control plane advertises for one credential; `page_size` splits the profiles."""

    active_profiles: tuple[str, ...] = ()
    inactive_profiles: tuple[str, ...] = ()
    on_demand_models: tuple[str, ...] = ()
    page_size: int | None = None

    def vendor_ids(self) -> frozenset[str]:
        return frozenset((*self.active_profiles, *self.on_demand_models))

    def invocable_ids(self) -> frozenset[str]:
        return frozenset(f"bedrock/{model_id}" for model_id in self.vendor_ids())

    def pages(self) -> tuple[tuple[Mapping[str, JsonValue], ...], ...]:
        summaries: Final[tuple[Mapping[str, JsonValue], ...]] = (
            *({"inferenceProfileId": name, "status": "ACTIVE"} for name in self.active_profiles),
            *({"inferenceProfileId": name, "status": "INACTIVE"} for name in self.inactive_profiles),
        )
        size: Final = len(summaries) if self.page_size is None else self.page_size
        return tuple(summaries[start : start + size] for start in range(0, len(summaries), size)) or ((),)

    def respond(self, request: ControlPlaneRequest) -> Reply:
        if request.method != "GET":
            return json_reply(405, {"message": f"{request.method} is not a listing"})
        if request.path == FOUNDATION_MODELS:
            if request.query != {"byInferenceType": "ON_DEMAND"}:
                return json_reply(400, {"message": f"unexpected foundation-models query {dict(request.query)}"})
            return json_reply(200, {"modelSummaries": [{"modelId": name} for name in self.on_demand_models]})
        if request.path != INFERENCE_PROFILES:
            return json_reply(404, {"message": f"no scripted listing at {request.path}"})
        pages: Final = self.pages()
        index: Final = int(request.query.get("nextToken", "page-0").removeprefix("page-"))
        expected_query: Final = {"maxResults": "1000", "typeEquals": "SYSTEM_DEFINED"} | (
            {"nextToken": f"page-{index}"} if index else {}
        )
        if dict(request.query) != expected_query or index >= len(pages):
            return json_reply(400, {"message": f"unexpected inference-profiles query {dict(request.query)}"})
        continuation: Final = {"nextToken": f"page-{index + 1}"} if index + 1 < len(pages) else {}
        return json_reply(200, {"inferenceProfileSummaries": list(pages[index]), **continuation})


@dataclass(frozen=True, slots=True)
class ControlPlane:
    """An owned HTTP CONNECT proxy terminating TLS for the hosted Bedrock control-plane names.

    The proxy under test gets it as ``HTTPS_PROXY`` and trusts its certificate through ``SSL_VERIFY``, so the
    lister's own ``https://bedrock.<region>.<suffix>`` URLs reach a scripted catalog without a DNS override.
    A CONNECT to any other host is refused with 403 and recorded in `refused`."""

    url: str
    certificate: Path
    received: SimpleQueue[ControlPlaneRequest]
    refused: SimpleQueue[str]
    responders: dict[str, Responder]  # mutable-ok: answering() adds a responder per test and removes it

    def environment(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "HTTPS_PROXY": self.url,
                "https_proxy": self.url,
                "NO_PROXY": NO_PROXY_HOSTS,
                "no_proxy": NO_PROXY_HOSTS,
                "SSL_VERIFY": str(self.certificate),
            }
        )

    def drain(self) -> tuple[ControlPlaneRequest, ...]:
        return tuple(self.received.get_nowait() for _ in range(self.received.qsize()))

    def refusals(self) -> tuple[str, ...]:
        return tuple(self.refused.get_nowait() for _ in range(self.refused.qsize()))

    @contextmanager
    def answering(self, credential: str, respond: Responder) -> Iterator[None]:
        self.responders[credential] = respond
        try:
            yield
        finally:
            del self.responders[credential]


def _header_map(handler: BaseHTTPRequestHandler) -> Mapping[str, str]:
    return MappingProxyType({name.lower(): value for name, value in handler.headers.items()})


def _answer(request: ControlPlaneRequest, responders: Mapping[str, Responder], errors: SimpleQueue[Exception]) -> Reply:
    try:
        return responders.get(request.credential, _unregistered)(request)
    except Exception as error:
        errors.put(error)
        return Reply(status=500)


def _tunneled_handler(
    host: str,
    received: SimpleQueue[ControlPlaneRequest],
    responders: Mapping[str, Responder],
    errors: SimpleQueue[Exception],
) -> type[BaseHTTPRequestHandler]:
    class Tunneled(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 60

        def respond(self) -> None:
            request: Final = ControlPlaneRequest(
                host,
                self.command,
                self.path,
                _header_map(self),
                self.rfile.read(int(self.headers.get("content-length", "0"))),
            )
            received.put(request)
            reply: Final = _answer(request, responders, errors)
            if reply.drop_connection:
                self.close_connection = True
                return
            try:
                self.send_response(reply.status)
                self.send_header("content-type", reply.content_type)
                for name, value in reply.headers.items():
                    self.send_header(name, value)
                self.send_header("content-length", str(len(reply.body)))
                self.end_headers()
                self.wfile.write(reply.body)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                self.close_connection = True

        do_GET = respond
        do_POST = respond

        def log_message(self, format: str, *args: object) -> None:
            pass

    return Tunneled


@contextmanager
def control_plane(directory: Path, hosts: tuple[str, ...]) -> Generator[ControlPlane, None, None]:
    cert_file, key_file = tls.write_self_signed_cert(directory, names=hosts)
    context: Final = tls.server_context(cert_file, key_file)
    trusted: Final = frozenset(hosts)
    received: Final[SimpleQueue[ControlPlaneRequest]] = SimpleQueue()
    refused: Final[SimpleQueue[str]] = SimpleQueue()
    errors: Final[SimpleQueue[Exception]] = SimpleQueue()
    responders: Final[dict[str, Responder]] = {}  # mutable-ok: cells register and remove their catalogs

    class Proxy(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 60

        def do_CONNECT(self) -> None:
            host: Final = self.path.rsplit(":", 1)[0]
            self.close_connection = True
            if host not in trusted:
                refused.put(self.path)
                self.send_response(403)
                self.send_header("content-length", "0")
                self.end_headers()
                return
            self.send_response(200, "Connection established")
            self.end_headers()
            self.wfile.flush()
            try:
                secured: Final = context.wrap_socket(self.connection, server_side=True)
            except OSError as error:
                errors.put(error)
                return
            with secured:
                _tunneled_handler(host, received, responders, errors)(secured, self.client_address, self.server)

        def log_message(self, format: str, *args: object) -> None:
            pass

    class Server(ThreadingHTTPServer):
        daemon_threads = True
        block_on_close = False
        request_queue_size = 128

    with Server(("127.0.0.1", 0), Proxy) as server:
        thread: Final = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
        thread.start()
        try:
            yield ControlPlane(f"http://127.0.0.1:{server.server_port}", cert_file, received, refused, responders)
        finally:
            server.shutdown()
            thread.join(timeout=6)
            assert not thread.is_alive(), "Owned control plane survived cleanup"
            server.server_close()
            failure: Final = None if errors.empty() else errors.get_nowait()
            assert failure is None, f"Owned control plane failed: {failure!r}"
