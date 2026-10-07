from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum, auto
from functools import cache
from itertools import accumulate
from typing import Final

import httpx
from typing_extensions import assert_never

from litellm.exceptions import APIConnectionError, Timeout


class TransportErrorKind(Enum):
    TIMEOUT = auto()
    DROPPED_CONNECTION = auto()


@dataclass(frozen=True, slots=True)
class _TransportClasses:
    timeout: tuple[type[BaseException], ...]
    dropped_connection: tuple[type[BaseException], ...]


_HTTPX: Final = _TransportClasses(
    timeout=(httpx.TimeoutException,),
    dropped_connection=(httpx.ReadError, httpx.RemoteProtocolError),
)
_MAX_CAUSE_DEPTH: Final = 20


@cache
def _transport_classes() -> _TransportClasses:
    try:
        from httpx2 import ReadError, RemoteProtocolError, TimeoutException
    except ImportError:
        return _HTTPX
    return _TransportClasses(
        timeout=(*_HTTPX.timeout, TimeoutException),
        dropped_connection=(*_HTTPX.dropped_connection, ReadError, RemoteProtocolError),
    )


def _next_cause(cause: BaseException | None, _: int) -> BaseException | None:
    return None if cause is None else cause.__cause__


def _explicit_causes(exc: BaseException) -> Iterator[BaseException]:
    chain: Final = accumulate(range(_MAX_CAUSE_DEPTH - 1), _next_cause, initial=exc)
    return (cause for cause in chain if cause is not None)


def classify_transport_error(exc: BaseException) -> TransportErrorKind | None:
    """The transport failure behind ``exc``, whichever of httpx or httpx2 raised it.

    Walks the explicit ``raise ... from`` chain, so an SDK wrapper such as
    ``openai.APIConnectionError`` classifies as the error it was raised from;
    implicit ``__context__`` is left alone because it is accidental.
    """
    classes: Final = _transport_classes()
    for cause in _explicit_causes(exc):
        if isinstance(cause, classes.timeout):
            return TransportErrorKind.TIMEOUT
        if isinstance(cause, classes.dropped_connection):
            return TransportErrorKind.DROPPED_CONNECTION
    return None


def as_public_exception(exc: Exception, model: str | None, llm_provider: str | None) -> Exception | None:
    kind: Final = classify_transport_error(exc)
    match kind:
        case TransportErrorKind.TIMEOUT:
            return Timeout(
                message=f"timed out reading the response: {exc}",
                model=model,
                llm_provider=llm_provider,
            )
        case TransportErrorKind.DROPPED_CONNECTION:
            return APIConnectionError(
                message=f"connection dropped while reading the response: {exc}",
                llm_provider=llm_provider,
                model=model,
            )
        case None:
            return None
    return assert_never(kind)
