from __future__ import annotations

import base64
import contextlib
import io
import json
import socket
import socketserver
import threading
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final, Literal
from urllib.parse import parse_qs, urlsplit

from integration._support import responses_vendor as rv
from integration._support.tls import server_context, write_self_signed_cert
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

CHATGPT_AUTH_HOST: Final = "auth.openai.com"
GITHUB_HOST: Final = "github.com"
GITHUB_API_HOST: Final = "api.github.com"
AUTH_HOSTS: Final = (CHATGPT_AUTH_HOST, GITHUB_HOST, GITHUB_API_HOST)
CHATGPT_ACCOUNT: Final = "acct-device-login"
_FAR_FUTURE: Final = 4102444800
_LONG_AGO: Final = 946684800
_OPENAI_AUTH_CLAIM: Final = "https://api.openai.com/auth"

CHATGPT_REFUSAL: Final = (
    "ChatGPT device-code login needs a human and cannot run inside a running event loop "
    "or a worker thread (for example the LiteLLM proxy). Log in once outside the proxy with "
    '`python -c "from litellm.llms.chatgpt.authenticator import Authenticator; '
    'Authenticator().get_access_token()"` and mount the resulting auth.json into the proxy, '
    "or set CHATGPT_TOKEN_DIR to a directory that already holds it."
)
COPILOT_REFUSAL: Final = (
    "GitHub Copilot device-code login needs a human and cannot run inside a running event loop "
    "or a worker thread (for example the LiteLLM proxy). Log in once outside the proxy with "
    '`python -c "from litellm.llms.github_copilot.authenticator import Authenticator; '
    'Authenticator().get_access_token()"` and mount the resulting access-token file into '
    "the proxy, or set GITHUB_COPILOT_TOKEN_DIR to a directory that already holds it."
)


def _segment(value: Mapping[str, JsonValue]) -> str:
    return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()


def chatgpt_jwt(subject: str, expires_at: int = _FAR_FUTURE) -> str:
    claims: Final[Mapping[str, JsonValue]] = {
        "sub": subject,
        "exp": expires_at,
        _OPENAI_AUTH_CLAIM: {"chatgpt_account_id": CHATGPT_ACCOUNT},
    }
    return f"{_segment({'alg': 'none', 'typ': 'JWT'})}.{_segment(claims)}.synthetic"


CHATGPT_STORED: Final = chatgpt_jwt("stored")
CHATGPT_REFRESHED: Final = chatgpt_jwt("refreshed")
CHATGPT_FIRST_LOGIN: Final = chatgpt_jwt("first-login")
CHATGPT_REJECTED: Final = chatgpt_jwt("rejected")
CHATGPT_EXPIRED: Final = chatgpt_jwt("expired", _LONG_AGO)
CHATGPT_ID_TOKEN: Final = chatgpt_jwt("identity")
GOOD_REFRESH: Final = "device-login-refresh-good"
REVOKED_REFRESH: Final = "device-login-refresh-revoked"
CHATGPT_USER_CODE: Final = "CGPT-LEAK"
COPILOT_USER_CODE: Final = "COPI-LEAK"
COPILOT_ACCESS: Final = "device-login-github-access"
COPILOT_REJECTED_ACCESS: Final = "device-login-github-access-rejected"
COPILOT_FIRST_LOGIN_ACCESS: Final = "device-login-github-access-first-login"
COPILOT_STORED_KEY: Final = "tid=device-login;stored"
COPILOT_MINTED_KEY: Final = "tid=device-login;minted"
COPILOT_REJECTED_KEY: Final = "tid=device-login;rejected"
CONTROL_KEY: Final = "device-login-control-key"
ACCEPTED_BEARERS: Final = frozenset(
    {CHATGPT_STORED, CHATGPT_REFRESHED, CHATGPT_FIRST_LOGIN, COPILOT_STORED_KEY, COPILOT_MINTED_KEY, CONTROL_KEY}
)
REJECTED_DETAIL: Final = "the scripted provider rejected this bearer token"
_DEVICE_AUTH_ID: Final = "device-auth-synthetic"
_AUTHORIZATION_CODE: Final = "authorization-code-synthetic"
_CODE_VERIFIER: Final = "code-verifier-synthetic"
_GITHUB_DEVICE_CODE: Final = "github-device-code-synthetic"
_EMBEDDING: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.5, 0.75]}],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 3, "total_tokens": 3},
    }
)


def _json(status: int, body: Mapping[str, JsonValue]) -> Reply:
    return Reply(status=status, body=json.dumps(dict(body)).encode())


def bearer(request: Request) -> str:
    return request.headers.get("authorization", "").removeprefix("Bearer ")


def _provider_api(vendor: rv.ResponsesVendor) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if bearer(request) not in ACCEPTED_BEARERS:
            return _json(401, {"detail": REJECTED_DETAIL})
        if urlsplit(request.target).path.endswith("/embeddings"):
            return _json(200, _EMBEDDING)
        return vendor.respond(request)

    return respond


def _chatgpt_tokens(access_token: str) -> Mapping[str, JsonValue]:
    return {"access_token": access_token, "id_token": CHATGPT_ID_TOKEN, "refresh_token": GOOD_REFRESH}


def _oauth_token(request: Request, grant: threading.Event) -> Reply:
    if request.headers.get("content-type", "").startswith("application/x-www-form-urlencoded"):
        form: Final = parse_qs(request.body.decode())
        exchanged: Final = form.get("code") == [_AUTHORIZATION_CODE] and form.get("code_verifier") == [_CODE_VERIFIER]
        if grant.is_set() and exchanged:
            return _json(200, _chatgpt_tokens(CHATGPT_FIRST_LOGIN))
        return _json(400, {"error": "invalid_grant"})
    body: Final = rv.JSON_OBJECT.validate_json(request.body)
    if body.get("grant_type") == "refresh_token" and body.get("refresh_token") == GOOD_REFRESH:
        return _json(200, _chatgpt_tokens(CHATGPT_REFRESHED))
    return _json(400, {"error": "invalid_grant", "error_description": "refresh token was revoked"})


def _copilot_api_key(request: Request, api_url: str) -> Reply:
    if request.headers.get("authorization") not in (f"token {COPILOT_ACCESS}", f"token {COPILOT_FIRST_LOGIN_ACCESS}"):
        return _json(401, {"message": "Bad credentials"})
    return _json(
        200,
        {"token": COPILOT_MINTED_KEY, "expires_at": int(time.time()) + 1800, "endpoints": {"api": api_url}},
    )


def _switched_off() -> Reply:
    return _json(503, {"error": "device login is switched off in this scenario"})


def _auth_hosts(api_url: str, grant: threading.Event) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        match (request.headers.get("host", ""), request.method, urlsplit(request.target).path):
            case ("auth.openai.com", "POST", "/oauth/token"):
                return _oauth_token(request, grant)
            case ("auth.openai.com", "POST", "/api/accounts/deviceauth/usercode") if grant.is_set():
                return _json(200, {"device_auth_id": _DEVICE_AUTH_ID, "user_code": CHATGPT_USER_CODE, "interval": "5"})
            case ("auth.openai.com", "POST", "/api/accounts/deviceauth/token") if grant.is_set():
                return _json(
                    200,
                    {
                        "authorization_code": _AUTHORIZATION_CODE,
                        "code_challenge": "code-challenge-synthetic",
                        "code_verifier": _CODE_VERIFIER,
                    },
                )
            case ("github.com", "POST", "/login/device/code") if grant.is_set():
                return _json(
                    200,
                    {
                        "device_code": _GITHUB_DEVICE_CODE,
                        "user_code": COPILOT_USER_CODE,
                        "verification_uri": "https://github.com/login/device",
                        "expires_in": 900,
                        "interval": 5,
                    },
                )
            case ("github.com", "POST", "/login/oauth/access_token") if grant.is_set():
                return _json(200, {"access_token": COPILOT_FIRST_LOGIN_ACCESS, "token_type": "bearer"})
            case ("api.github.com", "GET", "/copilot_internal/v2/token"):
                return _copilot_api_key(request, api_url)
            case _:
                return _switched_off()

    return respond


@dataclass(frozen=True, slots=True)
class Hangup:
    authority: str
    seconds: float


@dataclass(frozen=True, slots=True)
class Switches:
    grant: threading.Event
    hold_connect: threading.Event
    stall_tls: threading.Event
    refuse_connect: threading.Event

    def clear(self) -> None:
        for switch in (self.grant, self.hold_connect, self.stall_tls, self.refuse_connect):
            switch.clear()


@dataclass(frozen=True, slots=True)
class Peers:
    api: Wire
    auth: Wire
    tunnel: str
    cert: Path
    authorities: SimpleQueue[str]
    hangups: SimpleQueue[Hangup]
    switches: Switches

    def environment(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "HTTPS_PROXY": self.tunnel,
                "NO_PROXY": "127.0.0.1,localhost",
                "SSL_CERT_FILE": str(self.cert),
                "CHATGPT_API_BASE": self.api.url,
                "GITHUB_COPILOT_API_BASE": self.api.url,
            }
        )

    def auth_connections(self) -> tuple[str, ...]:
        return tuple(self.authorities.get_nowait() for _ in range(self.authorities.qsize()))

    def dropped(self) -> tuple[Hangup, ...]:
        return tuple(self.hangups.get_nowait() for _ in range(self.hangups.qsize()))

    def reset(self) -> None:
        self.switches.clear()
        self.api.drain()
        self.auth.drain()
        self.auth_connections()
        self.dropped()


def _pipe(source: socket.socket, sink: socket.socket) -> None:
    with contextlib.suppress(OSError):
        for chunk in iter(lambda: source.recv(65536), b""):
            sink.sendall(chunk)
    with contextlib.suppress(OSError):
        sink.shutdown(socket.SHUT_WR)


@dataclass(frozen=True, slots=True)
class _Held:
    outcome: Literal["released", "hung-up", "cut"]
    early_bytes: bytes


def _hold(connection: socket.socket, held: threading.Event, cut: threading.Event) -> _Held:
    received: Final = io.BytesIO()
    connection.settimeout(0.1)
    while held.is_set() and not cut.is_set():
        try:
            chunk: Final = connection.recv(65536)
        except TimeoutError:
            continue
        except OSError:
            return _Held("hung-up", b"")
        if chunk == b"":
            return _Held("hung-up", b"")
        received.write(chunk)
    if cut.is_set():
        return _Held("cut", b"")
    return _Held("released", received.getvalue())


class _TunnelServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 128


@contextmanager
def _tunnel(
    destination: Wire, switches: Switches, authorities: SimpleQueue[str], hangups: SimpleQueue[Hangup]
) -> Generator[str]:
    destination_port: Final = int(destination.url.rsplit(":", 1)[1])

    class Tunnel(socketserver.StreamRequestHandler):
        rbufsize = 0
        request: socket.socket

        def hold(self, authority: str, opened: float, switch: threading.Event) -> bytes | None:
            held: Final = _hold(self.request, switch, switches.refuse_connect)
            match held.outcome:
                case "released":
                    return held.early_bytes
                case "hung-up":
                    hangups.put(Hangup(authority, time.monotonic() - opened))
                    return None
                case "cut":
                    return None

        def handle(self) -> None:
            opened: Final = time.monotonic()
            request_line: Final = self.rfile.readline().decode().split()
            while self.rfile.readline() not in (b"\r\n", b""):
                pass
            if len(request_line) < 2:
                return
            authority: Final = request_line[1]
            if authority.rsplit(":", 1)[0] not in AUTH_HOSTS:
                self.wfile.write(b"HTTP/1.1 403 Forbidden\r\ncontent-length: 0\r\n\r\n")
                return
            authorities.put(authority)
            if switches.refuse_connect.is_set():
                self.wfile.write(b"HTTP/1.1 502 Bad Gateway\r\ncontent-length: 0\r\n\r\n")
                return
            if self.hold(authority, opened, switches.hold_connect) is None:
                return
            self.wfile.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            client_hello: Final = self.hold(authority, opened, switches.stall_tls)
            if client_hello is None:
                return
            self.request.settimeout(10)
            with socket.create_connection(("127.0.0.1", destination_port), timeout=10) as upstream:
                upstream.sendall(client_hello)
                outbound: Final = threading.Thread(target=_pipe, args=(self.request, upstream))
                outbound.start()
                _pipe(upstream, self.request)
                outbound.join(timeout=12)

    with _TunnelServer(("127.0.0.1", 0), Tunnel) as server:
        thread: Final = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            switches.clear()
            server.shutdown()
            thread.join(timeout=6)


@contextmanager
def device_login_peers(directory: Path) -> Generator[Peers]:
    cert, key = write_self_signed_cert(directory, AUTH_HOSTS)
    switches: Final = Switches(threading.Event(), threading.Event(), threading.Event(), threading.Event())
    authorities: Final[SimpleQueue[str]] = SimpleQueue()
    hangups: Final[SimpleQueue[Hangup]] = SimpleQueue()
    with (
        wire_server(_provider_api(rv.ResponsesVendor())) as api,
        wire_server(_auth_hosts(api.url, switches.grant), tls=server_context(cert, key)) as auth,
        _tunnel(auth, switches, authorities, hangups) as tunnel,
    ):
        yield Peers(api, auth, tunnel, cert, authorities, hangups, switches)


def chatgpt_record(
    access_token: str, *, refresh_token: str | None = None, expires_at: JsonValue = None
) -> Mapping[str, JsonValue]:
    optional: Final[Mapping[str, JsonValue]] = MappingProxyType(
        {"refresh_token": refresh_token, "expires_at": expires_at}
    )
    return MappingProxyType(
        {
            "access_token": access_token,
            "id_token": CHATGPT_ID_TOKEN,
            "account_id": CHATGPT_ACCOUNT,
            **{key: value for key, value in optional.items() if value is not None},
        }
    )


def write_chatgpt(directory: Path, record: Mapping[str, JsonValue]) -> None:
    (directory / "auth.json").write_text(json.dumps(dict(record)))


def write_copilot_key(directory: Path, token: str, api_url: str, expires_in: float = 3600) -> None:
    (directory / "api-key.json").write_text(
        json.dumps({"token": token, "expires_at": time.time() + expires_in, "endpoints": {"api": api_url}})
    )


def write_copilot_access(directory: Path, access_token: str) -> None:
    (directory / "access-token").write_text(access_token)


def clear_tokens(*directories: Path) -> None:
    for directory in directories:
        for path in directory.iterdir():
            path.unlink()
