from __future__ import annotations

import typing

from .imports import lazy_import
from .version import version as __version__  # noqa: F401


__all__ = [
    # .client
    "ClientProtocol",
    # .datastructures
    "Headers",
    "HeadersLike",
    "MultipleValuesError",
    # .exceptions
    "ConcurrencyError",
    "ConnectionClosed",
    "ConnectionClosedError",
    "ConnectionClosedOK",
    "DuplicateParameter",
    "InvalidHandshake",
    "InvalidHeader",
    "InvalidHeaderFormat",
    "InvalidHeaderValue",
    "InvalidOrigin",
    "InvalidParameterName",
    "InvalidParameterValue",
    "InvalidState",
    "InvalidStatus",
    "InvalidUpgrade",
    "InvalidURI",
    "NegotiationError",
    "PayloadTooBig",
    "ProtocolError",
    "SecurityError",
    "WebSocketException",
    "WebSocketProtocolError",
    # .legacy.auth
    "BasicAuthWebSocketServerProtocol",
    "basic_auth_protocol_factory",
    # .legacy.client
    "WebSocketClientProtocol",
    "connect",
    "unix_connect",
    # .legacy.exceptions
    "AbortHandshake",
    "InvalidMessage",
    "InvalidStatusCode",
    "RedirectHandshake",
    # .legacy.protocol
    "WebSocketCommonProtocol",
    # .legacy.server
    "WebSocketServer",
    "WebSocketServerProtocol",
    "broadcast",
    "serve",
    "unix_serve",
    # .server
    "ServerProtocol",
    # .typing
    "Data",
    "ExtensionName",
    "ExtensionParameter",
    "LoggerLike",
    "StatusLike",
    "Origin",
    "Subprotocol",
]

# When type checking, import non-deprecated aliases eagerly. Else, import on demand.
if typing.TYPE_CHECKING:
    from .client import ClientProtocol
    from .datastructures import Headers, HeadersLike, MultipleValuesError
    from .exceptions import (
        ConcurrencyError,
        ConnectionClosed,
        ConnectionClosedError,
        ConnectionClosedOK,
        DuplicateParameter,
        InvalidHandshake,
        InvalidHeader,
        InvalidHeaderFormat,
        InvalidHeaderValue,
        InvalidOrigin,
        InvalidParameterName,
        InvalidParameterValue,
        InvalidState,
        InvalidStatus,
        InvalidUpgrade,
        InvalidURI,
        NegotiationError,
        PayloadTooBig,
        ProtocolError,
        SecurityError,
        WebSocketException,
        WebSocketProtocolError,
    )
    from .legacy.auth import (
        BasicAuthWebSocketServerProtocol,
        basic_auth_protocol_factory,
    )
    from .legacy.client import WebSocketClientProtocol, connect, unix_connect
    from .legacy.exceptions import (
        AbortHandshake,
        InvalidMessage,
        InvalidStatusCode,
        RedirectHandshake,
    )
    from .legacy.protocol import WebSocketCommonProtocol
    from .legacy.server import (
        WebSocketServer,
        WebSocketServerProtocol,
        broadcast,
        serve,
        unix_serve,
    )
    from .server import ServerProtocol
    from .typing import (
        Data,
        ExtensionName,
        ExtensionParameter,
        LoggerLike,
        Origin,
        StatusLike,
        Subprotocol,
    )
else:
    lazy_import(
        globals(),
        aliases={
            # .client
            "ClientProtocol": ".client",
            # .datastructures
            "Headers": ".datastructures",
            "HeadersLike": ".datastructures",
            "MultipleValuesError": ".datastructures",
            # .exceptions
            "ConcurrencyError": ".exceptions",
            "ConnectionClosed": ".exceptions",
            "ConnectionClosedError": ".exceptions",
            "ConnectionClosedOK": ".exceptions",
            "DuplicateParameter": ".exceptions",
            "InvalidHandshake": ".exceptions",
            "InvalidHeader": ".exceptions",
            "InvalidHeaderFormat": ".exceptions",
            "InvalidHeaderValue": ".exceptions",
            "InvalidOrigin": ".exceptions",
            "InvalidParameterName": ".exceptions",
            "InvalidParameterValue": ".exceptions",
            "InvalidState": ".exceptions",
            "InvalidStatus": ".exceptions",
            "InvalidUpgrade": ".exceptions",
            "InvalidURI": ".exceptions",
            "NegotiationError": ".exceptions",
            "PayloadTooBig": ".exceptions",
            "ProtocolError": ".exceptions",
            "SecurityError": ".exceptions",
            "WebSocketException": ".exceptions",
            "WebSocketProtocolError": ".exceptions",
            # .legacy.auth
            "BasicAuthWebSocketServerProtocol": ".legacy.auth",
            "basic_auth_protocol_factory": ".legacy.auth",
            # .legacy.client
            "WebSocketClientProtocol": ".legacy.client",
            "connect": ".legacy.client",
            "unix_connect": ".legacy.client",
            # .legacy.exceptions
            "AbortHandshake": ".legacy.exceptions",
            "InvalidMessage": ".legacy.exceptions",
            "InvalidStatusCode": ".legacy.exceptions",
            "RedirectHandshake": ".legacy.exceptions",
            # .legacy.protocol
            "WebSocketCommonProtocol": ".legacy.protocol",
            # .legacy.server
            "WebSocketServer": ".legacy.server",
            "WebSocketServerProtocol": ".legacy.server",
            "broadcast": ".legacy.server",
            "serve": ".legacy.server",
            "unix_serve": ".legacy.server",
            # .server
            "ServerProtocol": ".server",
            # .typing
            "Data": ".typing",
            "ExtensionName": ".typing",
            "ExtensionParameter": ".typing",
            "LoggerLike": ".typing",
            "Origin": ".typing",
            "StatusLike": ".typing",
            "Subprotocol": ".typing",
        },
        deprecated_aliases={
            # deprecated in 9.0 - 2021-09-01
            "framing": ".legacy",
            "handshake": ".legacy",
            "parse_uri": ".uri",
            "WebSocketURI": ".uri",
        },
    )
