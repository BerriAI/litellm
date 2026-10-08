"""A recording Postgres front: speaks enough of the protocol to capture the
password a client presents, then authenticates upstream to the real server as a
fixed test role and relays bytes both ways.

Used to observe the literal credential a client (Prisma query engine, psql,
psycopg) sends: it answers the client ``AuthenticationCleartextPassword`` and
records ``(user, password)``. Upstream it performs whatever handshake the real
server asks for (trust, cleartext, MD5, SCRAM-SHA-256) with the role's real
password, then relays everything after ReadyForQuery verbatim.

The front answers SSLRequest and GSSENCRequest with 'N', so it only fronts
clients that tolerate plaintext (libpq/prisma ``sslmode=prefer`` or unset).
"""

import asyncio
import base64
import hashlib
import hmac
import os
import socket
import struct
import threading
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final

PORT: Final = 0
_SSL_REQUEST_CODE: Final = 80877103
_GSSENC_REQUEST_CODE: Final = 80877104
_PROTOCOL_3: Final = 196608

AUTH_OK: Final = 0
AUTH_CLEARTEXT: Final = 3
AUTH_MD5: Final = 5
AUTH_SASL: Final = 10
AUTH_SASL_CONTINUE: Final = 11
AUTH_SASL_FINAL: Final = 12


def _free_port() -> int:
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        return int(reserve.getsockname()[1])


def _cstring(payload: bytes, offset: int) -> tuple[str, int]:
    end: Final = payload.index(b"\x00", offset)
    return payload[offset:end].decode(), end + 1


def parse_startup_parameters(payload: bytes) -> dict[str, str]:
    parameters: Final[dict[str, str]] = {}
    offset = 0  # rebind-ok: walking the key/value pair list
    while offset < len(payload) - 1:
        key, offset = _cstring(payload, offset)
        if not key:
            break
        value, offset = _cstring(payload, offset)
        parameters[key] = value
    return parameters


def _startup_message(parameters: Mapping[str, str], protocol: int = _PROTOCOL_3) -> bytes:
    body: Final = struct.pack("!i", protocol) + b"".join(
        key.encode() + b"\x00" + value.encode() + b"\x00" for key, value in parameters.items()
    ) + b"\x00"
    return struct.pack("!i", len(body) + 4) + body


def _message(kind: bytes, payload: bytes) -> bytes:
    return kind + struct.pack("!i", len(payload) + 4) + payload


async def _read_message(reader: asyncio.StreamReader) -> tuple[bytes, bytes]:
    kind: Final = await reader.readexactly(1)
    length: Final = struct.unpack("!i", await reader.readexactly(4))[0]
    return kind, await reader.readexactly(length - 4)


async def _read_startup(reader: asyncio.StreamReader) -> tuple[int, bytes]:
    length: Final = struct.unpack("!i", await reader.readexactly(4))[0]
    body: Final = await reader.readexactly(length - 4)
    return struct.unpack("!i", body[:4])[0], body[4:]


def _scram_client_final(
    password: str, user: str, server_first: bytes, client_first_bare: str, client_nonce: str
) -> bytes:
    fields: Final = dict(item.split("=", 1) for item in server_first.decode().split(","))
    nonce: Final = fields["r"]
    assert nonce.startswith(client_nonce), "server echoed a different client nonce"
    salt: Final = base64.b64decode(fields["s"])
    iterations: Final = int(fields["i"])
    salted: Final = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    client_key: Final = hmac.new(salted, b"Client Key", hashlib.sha256).digest()
    stored_key: Final = hashlib.sha256(client_key).digest()
    without_proof: Final = f"c=biws,r={nonce}"
    auth_message: Final = f"{client_first_bare},{server_first.decode()},{without_proof}"
    signature: Final = hmac.new(stored_key, auth_message.encode(), hashlib.sha256).digest()
    proof: Final = bytes(a ^ b for a, b in zip(client_key, signature))
    return f"{without_proof},p={base64.b64encode(proof).decode()}".encode()


@dataclass(frozen=True, slots=True)
class RecordedLogin:
    user: str
    password: str


class PostgresFront:
    """Listens on 127.0.0.1:<port>; records each client's (user, password);
    serves queries by relaying to the real upstream as ``upstream_user``."""

    def __init__(
        self,
        upstream_host: str,
        upstream_port: int,
        upstream_user: str,
        upstream_password: str,
        upstream_database: str | None = None,
    ) -> None:
        self.port: Final = _free_port()
        self._upstream: Final = (upstream_host, upstream_port)
        self._upstream_user: Final = upstream_user
        self._upstream_password: Final = upstream_password
        self._upstream_database: Final = upstream_database
        self.logins: Final[list[RecordedLogin]] = []
        self.query_bytes = 0
        self._loop: Final = asyncio.new_event_loop()
        self._ready: Final = threading.Event()
        self._thread: Final = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()
        assert self._ready.wait(10), "Postgres front did not start"

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(10)

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(asyncio.start_server(self._serve, "127.0.0.1", self.port))
        self._ready.set()
        self._loop.run_forever()

    async def _authenticate_upstream(
        self, server_reader: asyncio.StreamReader, server_writer: asyncio.StreamWriter
    ) -> tuple[bytes, ...]:
        """Answer upstream auth challenges; return the post-auth messages to replay to the client."""
        post_auth: Final[list[bytes]] = []
        sasl_bare = ""  # rebind-ok: SCRAM state carried across the two SASL rounds
        sasl_nonce = ""  # rebind-ok: same
        while True:
            kind, payload = await _read_message(server_reader)
            if kind == b"R":
                auth_type: Final = struct.unpack("!i", payload[:4])[0]
                if auth_type == AUTH_OK:
                    post_auth.append(_message(kind, payload))
                elif auth_type == AUTH_CLEARTEXT:
                    server_writer.write(_message(b"p", self._upstream_password.encode() + b"\x00"))
                    await server_writer.drain()
                elif auth_type == AUTH_MD5:
                    inner: Final = hashlib.md5(
                        (self._upstream_password + self._upstream_user).encode()
                    ).hexdigest()
                    outer: Final = hashlib.md5(inner.encode() + payload[4:8]).hexdigest()
                    server_writer.write(_message(b"p", f"md5{outer}".encode() + b"\x00"))
                    await server_writer.drain()
                elif auth_type == AUTH_SASL:
                    client_nonce: Final = base64.b64encode(os.urandom(18)).decode()
                    sasl_bare = f"n=,r={client_nonce}"
                    sasl_nonce = client_nonce
                    initial: Final = ("n,," + sasl_bare).encode()
                    sasl_response: Final = (
                        b"SCRAM-SHA-256\x00" + struct.pack("!i", len(initial)) + initial
                    )
                    server_writer.write(_message(b"p", sasl_response))
                    await server_writer.drain()
                elif auth_type == AUTH_SASL_CONTINUE:
                    final: Final = _scram_client_final(
                        self._upstream_password, self._upstream_user, payload[4:], sasl_bare, sasl_nonce
                    )
                    server_writer.write(_message(b"p", final))
                    await server_writer.drain()
                elif auth_type == AUTH_SASL_FINAL:
                    continue
                else:
                    raise RuntimeError(f"unsupported upstream auth request {auth_type}")
            elif kind in (b"S", b"K", b"N"):
                post_auth.append(_message(kind, payload))
            elif kind == b"Z":
                post_auth.append(_message(kind, payload))
                return tuple(post_auth)
            elif kind == b"E":
                post_auth.append(_message(kind, payload))
                return tuple(post_auth)
            else:
                post_auth.append(_message(kind, payload))

    async def _serve(self, client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
        parameters: Final[dict[str, str]] = {}
        try:
            while True:
                code, payload = await _read_startup(client_reader)
                if code in (_SSL_REQUEST_CODE, _GSSENC_REQUEST_CODE):
                    client_writer.write(b"N")
                    await client_writer.drain()
                    continue
                parameters.update(parse_startup_parameters(payload))
                break
        except (asyncio.IncompleteReadError, ConnectionError):
            client_writer.close()
            return

        client_writer.write(_message(b"R", struct.pack("!i", AUTH_CLEARTEXT)))
        await client_writer.drain()
        try:
            kind, password_payload = await _read_message(client_reader)
            assert kind == b"p", f"expected PasswordMessage, got {kind!r}"
        except (asyncio.IncompleteReadError, ConnectionError, AssertionError):
            client_writer.close()
            return
        password: Final = password_payload.rstrip(b"\x00").decode()
        self.logins.append(RecordedLogin(user=parameters.get("user", ""), password=password))

        try:
            server_reader, server_writer = await asyncio.open_connection(*self._upstream)
            upstream_params: Final = {
                **parameters,
                "user": self._upstream_user,
                **({"database": self._upstream_database} if self._upstream_database else {}),
            }
            server_writer.write(_startup_message(upstream_params))
            await server_writer.drain()
            post_auth: Final = await self._authenticate_upstream(server_reader, server_writer)
            client_writer.write(b"".join(post_auth))
            await client_writer.drain()
        except (OSError, asyncio.IncompleteReadError, ConnectionError, RuntimeError) as e:
            client_writer.write(
                _message(b"E", b"SFATAL\x00C28000\x00M" + str(e).encode() + b"\x00\x00")
            )
            client_writer.close()
            return

        async def forward(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, count: bool) -> None:
            try:
                while chunk := await reader.read(65536):
                    if count:
                        self.query_bytes += len(chunk)
                    writer.write(chunk)
                    await writer.drain()
            except (ConnectionError, asyncio.IncompleteReadError):
                return
            finally:
                writer.close()

        await asyncio.gather(
            forward(client_reader, server_writer, True),
            forward(server_reader, client_writer, False),
        )


@contextmanager
def postgres_front(
    upstream_host: str,
    upstream_port: int,
    upstream_user: str,
    upstream_password: str,
    upstream_database: str | None = None,
) -> Generator[PostgresFront]:
    front: Final = PostgresFront(upstream_host, upstream_port, upstream_user, upstream_password, upstream_database)
    front.start()
    try:
        yield front
    finally:
        front.stop()
