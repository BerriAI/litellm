import queue
import socket
import socketserver
import ssl
import struct
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Final

import pytest
from confluent_kafka import KafkaError, Producer

from litellm.integrations.kafka import KafkaLogger
from litellm.integrations.kafka.config import KafkaConfigValue
from litellm.types.integrations.kafka import KafkaInitParams
from tests.integration._support.tls import server_context, write_self_signed_cert

NOW: Final = datetime(2025, 1, 1, tzinfo=timezone.utc)
TOKEN: Final = b"\0test-user\0test-password"


class AuthenticationPeer(socketserver.ThreadingTCPServer):
    daemon_threads = True

    def __init__(self, context: ssl.SSLContext) -> None:
        self.context: Final = context
        self.tokens: Final[queue.Queue[bytes]] = queue.Queue()
        self.authenticated: Final = threading.Event()
        super().__init__(("127.0.0.1", 0), AuthenticationHandler)


class AuthenticationHandler(socketserver.BaseRequestHandler):
    request: socket.socket

    def handle(self) -> None:
        assert isinstance(self.server, AuthenticationPeer)
        self.request.settimeout(2)
        try:
            with self.server.context.wrap_socket(self.request, server_side=True) as connection:
                with connection.makefile("rwb") as stream:
                    self.exchange(stream)
        except (ssl.SSLError, OSError):
            pass

    def exchange(self, stream: BinaryIO) -> None:
        assert isinstance(self.server, AuthenticationPeer)
        while header := stream.read(4):
            packet: Final = stream.read(struct.unpack(">i", header)[0])
            api, version, correlation = struct.unpack(">hhi", packet[:8])
            offset: Final = 10 + max(0, struct.unpack(">h", packet[8:10])[0])
            if api == 18:
                versions: Final = ((0, 0, 3), (3, 0, 0), (17, 0, 1), (18, 0, 3), (22, 0, 0), (36, 0, 1))
                response: Final = (
                    struct.pack(">hB", 0, len(versions) + 1)
                    + b"".join(struct.pack(">hhhB", *entry, 0) for entry in versions)
                    + struct.pack(">iB", 0, 0)
                    if version >= 3
                    else struct.pack(">hi", 0, len(versions))
                    + b"".join(struct.pack(">hhh", *entry) for entry in versions)
                )
                self.reply(stream, correlation, response)
            elif api == 17:
                assert packet[offset:] == b"\0\x05PLAIN"
                self.reply(stream, correlation, struct.pack(">hih", 0, 1, 5) + b"PLAIN")
            elif api == 36:
                token: Final = packet[offset + 4 :]
                self.server.tokens.put(token)
                error: Final = 0 if token == TOKEN else 58
                self.reply(stream, correlation, struct.pack(">hhi", error, -1, 0) + struct.pack(">q", 0))
            else:
                self.server.authenticated.set()
                return

    @staticmethod
    def reply(stream: BinaryIO, correlation: int, body: bytes) -> None:
        stream.write(struct.pack(">ii", len(body) + 4, correlation) + body)
        stream.flush()


@pytest.mark.parametrize(
    ("certificate", "policy", "password", "expected"),
    (
        ("trusted", "https", "test-password", None),
        ("mismatch", "https", "test-password", KafkaError._SSL),
        ("mismatch", "none", "test-password", None),
        ("untrusted", "none", "test-password", KafkaError._SSL),
        ("trusted", "https", "wrong-password", KafkaError._AUTHENTICATION),
    ),
)
def test_native_tls_and_plain_authentication(
    tmp_path: Path, certificate: str, policy: str, password: str, expected: int | None
) -> None:
    cert, key = write_self_signed_cert(tmp_path, ("broker.invalid",) if certificate == "mismatch" else ("localhost",))
    untrusted: Final = tmp_path / "untrusted"
    untrusted.mkdir()
    other_cert, _ = write_self_signed_cert(untrusted)
    failed: Final = threading.Event()

    def report(error: KafkaError) -> None:
        if error.code() == expected:
            failed.set()

    def factory(settings: Mapping[str, KafkaConfigValue]) -> Producer:
        return Producer({**settings, "error_cb": report})

    with AuthenticationPeer(server_context(cert, key)) as peer:
        worker: Final = threading.Thread(target=peer.serve_forever, daemon=True)
        worker.start()
        logger: Final = KafkaLogger(
            KafkaInitParams.model_validate(
                {
                    "bootstrap_servers": f"localhost:{peer.server_address[1]}",
                    "topic": "events",
                    "security_protocol": "SASL_SSL",
                    "sasl_username": "test-user",
                    "sasl_password": password,
                    "ssl_ca_location": str(other_cert if certificate == "untrusted" else cert),
                    "ssl_endpoint_identification_algorithm": policy,
                    "request_timeout_ms": 500,
                    "delivery_timeout_ms": 1000,
                }
            ),
            factory,
        )
        try:
            logger.log_success_event({"standard_logging_object": {"id": "tls-event"}}, None, NOW, NOW)
            if expected is None:
                assert peer.authenticated.wait(5), "client did not finish SASL authentication over TLS"
                assert peer.tokens.get(timeout=1) == TOKEN
            else:
                assert failed.wait(5), "native client did not report the expected TLS or SASL failure"
                if expected == KafkaError._SSL:
                    assert peer.tokens.empty(), "client sent credentials before verifying the server certificate"
        finally:
            logger.close(0)
            peer.shutdown()
            worker.join(5)
