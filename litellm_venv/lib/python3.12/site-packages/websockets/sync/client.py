from __future__ import annotations

import socket
import ssl as ssl_module
import threading
import warnings
from typing import Any, Sequence

from ..client import ClientProtocol
from ..datastructures import HeadersLike
from ..extensions.base import ClientExtensionFactory
from ..extensions.permessage_deflate import enable_client_permessage_deflate
from ..headers import validate_subprotocols
from ..http11 import USER_AGENT, Response
from ..protocol import CONNECTING, Event
from ..typing import LoggerLike, Origin, Subprotocol
from ..uri import parse_uri
from .connection import Connection
from .utils import Deadline


__all__ = ["connect", "unix_connect", "ClientConnection"]


class ClientConnection(Connection):
    """
    :mod:`threading` implementation of a WebSocket client connection.

    :class:`ClientConnection` provides :meth:`recv` and :meth:`send` methods for
    receiving and sending messages.

    It supports iteration to receive messages::

        for message in websocket:
            process(message)

    The iterator exits normally when the connection is closed with close code
    1000 (OK) or 1001 (going away) or without a close code. It raises a
    :exc:`~websockets.exceptions.ConnectionClosedError` when the connection is
    closed with any other code.

    Args:
        socket: Socket connected to a WebSocket server.
        protocol: Sans-I/O connection.
        close_timeout: Timeout for closing the connection in seconds.

    """

    def __init__(
        self,
        socket: socket.socket,
        protocol: ClientProtocol,
        *,
        close_timeout: float | None = 10,
    ) -> None:
        self.protocol: ClientProtocol
        self.response_rcvd = threading.Event()
        super().__init__(
            socket,
            protocol,
            close_timeout=close_timeout,
        )

    def handshake(
        self,
        additional_headers: HeadersLike | None = None,
        user_agent_header: str | None = USER_AGENT,
        timeout: float | None = None,
    ) -> None:
        """
        Perform the opening handshake.

        """
        with self.send_context(expected_state=CONNECTING):
            self.request = self.protocol.connect()
            if additional_headers is not None:
                self.request.headers.update(additional_headers)
            if user_agent_header is not None:
                self.request.headers["User-Agent"] = user_agent_header
            self.protocol.send_request(self.request)

        if not self.response_rcvd.wait(timeout):
            raise TimeoutError("timed out during handshake")

        # self.protocol.handshake_exc is always set when the connection is lost
        # before receiving a response, when the response cannot be parsed, or
        # when the response fails the handshake.

        if self.protocol.handshake_exc is not None:
            raise self.protocol.handshake_exc

    def process_event(self, event: Event) -> None:
        """
        Process one incoming event.

        """
        # First event - handshake response.
        if self.response is None:
            assert isinstance(event, Response)
            self.response = event
            self.response_rcvd.set()
        # Later events - frames.
        else:
            super().process_event(event)

    def recv_events(self) -> None:
        """
        Read incoming data from the socket and process events.

        """
        try:
            super().recv_events()
        finally:
            # If the connection is closed during the handshake, unblock it.
            self.response_rcvd.set()


def connect(
    uri: str,
    *,
    # TCP/TLS
    sock: socket.socket | None = None,
    ssl: ssl_module.SSLContext | None = None,
    server_hostname: str | None = None,
    # WebSocket
    origin: Origin | None = None,
    extensions: Sequence[ClientExtensionFactory] | None = None,
    subprotocols: Sequence[Subprotocol] | None = None,
    additional_headers: HeadersLike | None = None,
    user_agent_header: str | None = USER_AGENT,
    compression: str | None = "deflate",
    # Timeouts
    open_timeout: float | None = 10,
    close_timeout: float | None = 10,
    # Limits
    max_size: int | None = 2**20,
    # Logging
    logger: LoggerLike | None = None,
    # Escape hatch for advanced customization
    create_connection: type[ClientConnection] | None = None,
    **kwargs: Any,
) -> ClientConnection:
    """
    Connect to the WebSocket server at ``uri``.

    This function returns a :class:`ClientConnection` instance, which you can
    use to send and receive messages.

    :func:`connect` may be used as a context manager::

        from websockets.sync.client import connect

        with connect(...) as websocket:
            ...

    The connection is closed automatically when exiting the context.

    Args:
        uri: URI of the WebSocket server.
        sock: Preexisting TCP socket. ``sock`` overrides the host and port
            from ``uri``. You may call :func:`socket.create_connection` to
            create a suitable TCP socket.
        ssl: Configuration for enabling TLS on the connection.
        server_hostname: Host name for the TLS handshake. ``server_hostname``
            overrides the host name from ``uri``.
        origin: Value of the ``Origin`` header, for servers that require it.
        extensions: List of supported extensions, in order in which they
            should be negotiated and run.
        subprotocols: List of supported subprotocols, in order of decreasing
            preference.
        additional_headers (HeadersLike | None): Arbitrary HTTP headers to add
            to the handshake request.
        user_agent_header: Value of  the ``User-Agent`` request header.
            It defaults to ``"Python/x.y.z websockets/X.Y"``.
            Setting it to :obj:`None` removes the header.
        compression: The "permessage-deflate" extension is enabled by default.
            Set ``compression`` to :obj:`None` to disable it. See the
            :doc:`compression guide <../../topics/compression>` for details.
        open_timeout: Timeout for opening the connection in seconds.
            :obj:`None` disables the timeout.
        close_timeout: Timeout for closing the connection in seconds.
            :obj:`None` disables the timeout.
        max_size: Maximum size of incoming messages in bytes.
            :obj:`None` disables the limit.
        logger: Logger for this client.
            It defaults to ``logging.getLogger("websockets.client")``.
            See the :doc:`logging guide <../../topics/logging>` for details.
        create_connection: Factory for the :class:`ClientConnection` managing
            the connection. Set it to a wrapper or a subclass to customize
            connection handling.

    Any other keyword arguments are passed to :func:`~socket.create_connection`.

    Raises:
        InvalidURI: If ``uri`` isn't a valid WebSocket URI.
        OSError: If the TCP connection fails.
        InvalidHandshake: If the opening handshake fails.
        TimeoutError: If the opening handshake times out.

    """

    # Process parameters

    # Backwards compatibility: ssl used to be called ssl_context.
    if ssl is None and "ssl_context" in kwargs:
        ssl = kwargs.pop("ssl_context")
        warnings.warn(  # deprecated in 13.0 - 2024-08-20
            "ssl_context was renamed to ssl",
            DeprecationWarning,
        )

    wsuri = parse_uri(uri)
    if not wsuri.secure and ssl is not None:
        raise TypeError("ssl argument is incompatible with a ws:// URI")

    # Private APIs for unix_connect()
    unix: bool = kwargs.pop("unix", False)
    path: str | None = kwargs.pop("path", None)

    if unix:
        if path is None and sock is None:
            raise TypeError("missing path argument")
        elif path is not None and sock is not None:
            raise TypeError("path and sock arguments are incompatible")

    if subprotocols is not None:
        validate_subprotocols(subprotocols)

    if compression == "deflate":
        extensions = enable_client_permessage_deflate(extensions)
    elif compression is not None:
        raise ValueError(f"unsupported compression: {compression}")

    # Calculate timeouts on the TCP, TLS, and WebSocket handshakes.
    # The TCP and TLS timeouts must be set on the socket, then removed
    # to avoid conflicting with the WebSocket timeout in handshake().
    deadline = Deadline(open_timeout)

    if create_connection is None:
        create_connection = ClientConnection

    try:
        # Connect socket

        if sock is None:
            if unix:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.settimeout(deadline.timeout())
                assert path is not None  # mypy cannot figure this out
                sock.connect(path)
            else:
                kwargs.setdefault("timeout", deadline.timeout())
                sock = socket.create_connection((wsuri.host, wsuri.port), **kwargs)
            sock.settimeout(None)

        # Disable Nagle algorithm

        if not unix:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)

        # Initialize TLS wrapper and perform TLS handshake

        if wsuri.secure:
            if ssl is None:
                ssl = ssl_module.create_default_context()
            if server_hostname is None:
                server_hostname = wsuri.host
            sock.settimeout(deadline.timeout())
            sock = ssl.wrap_socket(sock, server_hostname=server_hostname)
            sock.settimeout(None)

        # Initialize WebSocket protocol

        protocol = ClientProtocol(
            wsuri,
            origin=origin,
            extensions=extensions,
            subprotocols=subprotocols,
            max_size=max_size,
            logger=logger,
        )

        # Initialize WebSocket connection

        connection = create_connection(
            sock,
            protocol,
            close_timeout=close_timeout,
        )
    except Exception:
        if sock is not None:
            sock.close()
        raise

    try:
        connection.handshake(
            additional_headers,
            user_agent_header,
            deadline.timeout(),
        )
    except Exception:
        connection.close_socket()
        connection.recv_events_thread.join()
        raise

    return connection


def unix_connect(
    path: str | None = None,
    uri: str | None = None,
    **kwargs: Any,
) -> ClientConnection:
    """
    Connect to a WebSocket server listening on a Unix socket.

    This function accepts the same keyword arguments as :func:`connect`.

    It's only available on Unix.

    It's mainly useful for debugging servers listening on Unix sockets.

    Args:
        path: File system path to the Unix socket.
        uri: URI of the WebSocket server. ``uri`` defaults to
            ``ws://localhost/`` or, when a ``ssl`` is provided, to
            ``wss://localhost/``.

    """
    if uri is None:
        # Backwards compatibility: ssl used to be called ssl_context.
        if kwargs.get("ssl") is None and kwargs.get("ssl_context") is None:
            uri = "ws://localhost/"
        else:
            uri = "wss://localhost/"
    return connect(uri=uri, unix=True, path=path, **kwargs)
