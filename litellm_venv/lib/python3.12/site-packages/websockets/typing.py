from __future__ import annotations

import http
import logging
import typing
from typing import Any, List, NewType, Optional, Tuple, Union


__all__ = [
    "Data",
    "LoggerLike",
    "StatusLike",
    "Origin",
    "Subprotocol",
    "ExtensionName",
    "ExtensionParameter",
]


# Public types used in the signature of public APIs

# Change to str | bytes when dropping Python < 3.10.
Data = Union[str, bytes]
"""Types supported in a WebSocket message:
:class:`str` for a Text_ frame, :class:`bytes` for a Binary_.

.. _Text: https://datatracker.ietf.org/doc/html/rfc6455#section-5.6
.. _Binary : https://datatracker.ietf.org/doc/html/rfc6455#section-5.6

"""


# Change to logging.Logger | ... when dropping Python < 3.10.
if typing.TYPE_CHECKING:
    LoggerLike = Union[logging.Logger, logging.LoggerAdapter[Any]]
    """Types accepted where a :class:`~logging.Logger` is expected."""
else:  # remove this branch when dropping support for Python < 3.11
    LoggerLike = Union[logging.Logger, logging.LoggerAdapter]
    """Types accepted where a :class:`~logging.Logger` is expected."""


# Change to http.HTTPStatus | int when dropping Python < 3.10.
StatusLike = Union[http.HTTPStatus, int]
"""
Types accepted where an :class:`~http.HTTPStatus` is expected."""


Origin = NewType("Origin", str)
"""Value of a ``Origin`` header."""


Subprotocol = NewType("Subprotocol", str)
"""Subprotocol in a ``Sec-WebSocket-Protocol`` header."""


ExtensionName = NewType("ExtensionName", str)
"""Name of a WebSocket extension."""

# Change to tuple[str, Optional[str]] when dropping Python < 3.9.
# Change to tuple[str, str | None] when dropping Python < 3.10.
ExtensionParameter = Tuple[str, Optional[str]]
"""Parameter of a WebSocket extension."""


# Private types

# Change to tuple[.., list[...]] when dropping Python < 3.9.
ExtensionHeader = Tuple[ExtensionName, List[ExtensionParameter]]
"""Extension in a ``Sec-WebSocket-Extensions`` header."""


ConnectionOption = NewType("ConnectionOption", str)
"""Connection option in a ``Connection`` header."""


UpgradeProtocol = NewType("UpgradeProtocol", str)
"""Upgrade protocol in an ``Upgrade`` header."""
