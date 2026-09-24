from __future__ import annotations

import socketserver
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from email import message_from_bytes
from email.message import Message
from queue import SimpleQueue
from typing import Final


@dataclass(frozen=True, slots=True)
class Delivery:
    sender: str
    recipients: tuple[str, ...]
    message: Message

    @property
    def subject(self) -> str:
        return str(self.message["Subject"])

    @property
    def html(self) -> str:
        for part in self.message.walk():
            if part.get_content_type() == "text/html":
                return part.get_payload(decode=True).decode()
        return ""


@dataclass(frozen=True, slots=True)
class Mailbox:
    host: str
    port: int
    received: SimpleQueue[Delivery]

    def pending(self) -> int:
        return self.received.qsize()

    def drain(self) -> tuple[Delivery, ...]:
        return tuple(self.received.get_nowait() for _ in range(self.received.qsize()))


def _address(argument: str) -> str:
    return argument.split(":", 1)[1].strip().strip("<>")


@contextmanager
def smtp_sink() -> Generator[Mailbox, None, None]:
    """Owned plaintext SMTP peer; deliveries traverse the proxy's real smtplib client."""
    received: Final[SimpleQueue[Delivery]] = SimpleQueue()
    errors: Final[SimpleQueue[Exception]] = SimpleQueue()

    class Handler(socketserver.StreamRequestHandler):
        timeout = 5

        def handle(self) -> None:
            try:
                self._session()
            except Exception as error:
                errors.put(error)

        def _reply(self, line: str) -> None:
            self.wfile.write(f"{line}\r\n".encode())
            self.wfile.flush()

        def _session(self) -> None:
            self._reply("220 integration-smtp ready")
            # rebind-ok: the SMTP envelope is built across MAIL/RCPT lines and reset after DATA or RSET.
            sender = ""
            recipients: tuple[str, ...] = ()
            while True:
                raw: Final = self.rfile.readline()
                if not raw:
                    return
                line: Final = raw.decode().rstrip("\r\n")
                verb: Final = line.split(" ", 1)[0].upper()
                if verb in {"EHLO", "HELO"}:
                    self._reply("250 integration-smtp")
                elif verb == "MAIL":
                    sender = _address(line)
                    self._reply("250 OK")
                elif verb == "RCPT":
                    recipients = (*recipients, _address(line))
                    self._reply("250 OK")
                elif verb == "DATA":
                    self._reply("354 End data with <CR><LF>.<CR><LF>")
                    body = bytearray()
                    while True:
                        chunk: Final = self.rfile.readline()
                        if not chunk or chunk == b".\r\n":
                            break
                        body.extend(chunk[1:] if chunk.startswith(b"..") else chunk)
                    received.put(Delivery(sender, recipients, message_from_bytes(bytes(body))))
                    sender, recipients = "", ()
                    self._reply("250 OK queued")
                elif verb == "RSET":
                    sender, recipients = "", ()
                    self._reply("250 OK")
                elif verb == "NOOP":
                    self._reply("250 OK")
                elif verb == "QUIT":
                    self._reply("221 Bye")
                    return
                else:
                    self._reply("502 Command not implemented")

    class OwnedServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = False

    with OwnedServer(("127.0.0.1", 0), Handler) as server:
        thread: Final = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
        thread.start()
        try:
            yield Mailbox("127.0.0.1", server.server_address[1], received)
        finally:
            server.shutdown()
            thread.join(timeout=6)
            assert not thread.is_alive(), "Owned SMTP server survived cleanup"
            server.server_close()
            failure: Final = None if errors.empty() else errors.get_nowait()
            assert failure is None, f"Owned SMTP peer failed: {failure!r}"
