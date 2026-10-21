import sys
from collections.abc import Iterator
from enum import Enum
from typing import Final, TypeAlias

import httpx


class TransportErrorKind(Enum):
    TIMEOUT = "timeout"
    DROPPED_CONNECTION = "dropped_connection"


_ErrorTable: TypeAlias = tuple[tuple[type[BaseException], TransportErrorKind], ...]

_HTTPX_ERRORS: Final[_ErrorTable] = (
    (httpx.TimeoutException, TransportErrorKind.TIMEOUT),
    (httpx.ReadError, TransportErrorKind.DROPPED_CONNECTION),
    (httpx.RemoteProtocolError, TransportErrorKind.DROPPED_CONNECTION),
)
_MAX_CAUSE_DEPTH: Final = 16


def _transport_errors() -> _ErrorTable:
    if "httpx2" not in sys.modules:
        return _HTTPX_ERRORS
    import httpx2

    return (
        *_HTTPX_ERRORS,
        (httpx2.TimeoutException, TransportErrorKind.TIMEOUT),
        (httpx2.ReadError, TransportErrorKind.DROPPED_CONNECTION),
        (httpx2.RemoteProtocolError, TransportErrorKind.DROPPED_CONNECTION),
    )


def transport_error_kind(error: object) -> TransportErrorKind | None:
    return next((kind for error_class, kind in _transport_errors() if isinstance(error, error_class)), None)


def transport_error_classes(kind: TransportErrorKind) -> tuple[type[BaseException], ...]:
    return tuple(error_class for error_class, error_kind in _transport_errors() if error_kind is kind)


def _explicit_causes(error: BaseException | None, depth: int) -> Iterator[BaseException]:
    if error is None or depth == 0:
        return
    yield error
    yield from _explicit_causes(error.__cause__, depth - 1)


def chained_transport_error_kind(error: BaseException) -> TransportErrorKind | None:
    kinds: Final = (transport_error_kind(cause) for cause in _explicit_causes(error, _MAX_CAUSE_DEPTH))
    return next((kind for kind in kinds if kind is not None), None)
