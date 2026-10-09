"""
Handles logging DB success/failure to ServiceLogger()

ServiceLogger() then sends DB logs to Prometheus, OTEL, Datadog etc
"""

import asyncio
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from types import MappingProxyType
from typing import Final

from litellm._logging import verbose_proxy_logger
from litellm._service_logger import ServiceTypes

_PRISMA_CLIENT_CRUD: Final = frozenset({"get_data", "update_data", "delete_data"})
_DEFAULT_TABLE_BY_KWARG: Final[Mapping[str, str]] = MappingProxyType(
    {"token": "key", "tokens": "key", "user_id": "user", "team_id": "team"}
)


def _table_metadata_reader(infers_table: bool) -> Callable[[Mapping[str, object]], dict[str, str] | None]:
    """Minimal, non-sensitive ``event_metadata`` for a DB service log.

    The raw ``kwargs``/``args`` carry live objects (Prisma client, OTel spans)
    and secrets (tokens), none of which belongs on a span, so only the table name
    surfaces. A ``PrismaClient`` CRUD method called without ``table_name`` picks
    its table from the lookup key, in the same order the method dispatches on.
    """

    def read(kwargs: Mapping[str, object]) -> dict[str, str] | None:
        table_name: Final = kwargs.get("table_name")
        if isinstance(table_name, str):
            return {"table_name": table_name}
        if not infers_table:
            return None
        inferred: Final = next(
            (table for key, table in _DEFAULT_TABLE_BY_KWARG.items() if kwargs.get(key) is not None), None
        )
        return {"table_name": inferred} if inferred is not None else None

    return read


class _DbIoWitness:
    """Activity an inner decorated call already reported stays with it; only unreported activity reaches the enclosing call."""

    __slots__ = ("_open", "_parent", "_reported", "_touched")

    def __init__(self, parent: "_DbIoWitness | None") -> None:
        self._parent: Final = parent
        self._touched = False
        self._reported = False
        self._open = True

    @property
    def touched(self) -> bool:
        return self._touched

    @property
    def open(self) -> bool:
        return self._open

    def mark(self) -> None:
        if self._open:
            self._touched = True

    def report(self) -> None:
        self._reported = True

    def close(self) -> None:
        self._open = False
        if self._touched and not self._reported and self._parent is not None:
            self._parent.mark()


_db_io_witness: Final[ContextVar["_DbIoWitness | None"]] = ContextVar("litellm_db_io_witness", default=None)


def record_db_io() -> None:
    witness: Final = _db_io_witness.get()
    if witness is not None:
        witness.mark()


def db_io_claimed() -> bool:
    """Whether an enclosing producer (``@log_db_metrics`` or ``db_span``) reports the Prisma I/O that runs now.

    A task spawned inside a producer inherits its witness by context copy; once that producer has
    finished, the copy is closed and the task's own Prisma I/O is nobody's to report but its own.
    """
    witness: Final = _db_io_witness.get()
    return witness is not None and witness.open


@contextmanager
def claim_db_io() -> Generator[_DbIoWitness]:
    """Own the DB event for the Prisma I/O inside: inner producers and the engine fallback stay quiet."""
    witness: Final = _DbIoWitness(parent=_db_io_witness.get())
    token: Final = _db_io_witness.set(witness)
    try:
        yield witness
    finally:
        witness.report()
        witness.close()
        _db_io_witness.reset(token)


def log_db_metrics(func):
    """
    Decorator to log the duration of a DB related function to ServiceLogger()

    Handles logging DB success/failure to ServiceLogger(), which logs to Prometheus, OTEL, Datadog

    When logging Failure it checks if the Exception is a PrismaError or an httpx.TransportError and then logs that as a DB Service Failure

    Args:
        func: The function to be decorated

    Returns:
        Result from the decorated function

    Raises:
        Exception: If the decorated function raises an exception
    """

    metadata_of: Final = _table_metadata_reader(func.__name__ in _PRISMA_CLIENT_CRUD)

    @wraps(func)
    async def wrapper(*args, **kwargs: object):
        start_time: Final[datetime] = datetime.now()
        witness: Final = _DbIoWitness(parent=_db_io_witness.get())
        witness_token: Final = _db_io_witness.set(witness)

        try:
            result: Final = await func(*args, **kwargs)
            end_time: datetime = datetime.now()
            from litellm.proxy.proxy_server import proxy_logging_obj

            if not witness.touched:
                return result
            asyncio.create_task(
                proxy_logging_obj.service_logging_obj.async_service_success_hook(
                    service=ServiceTypes.DB,
                    call_type=func.__name__,
                    parent_otel_span=kwargs.get("parent_otel_span", None),
                    duration=(end_time - start_time).total_seconds(),
                    start_time=start_time,
                    end_time=end_time,
                    event_metadata=metadata_of(kwargs),
                )
            )
            witness.report()
            return result
        except Exception as e:
            end_time: datetime = datetime.now()
            if await _handle_logging_db_exception(
                e=e,
                func=func,
                kwargs=kwargs,
                args=args,
                start_time=start_time,
                end_time=end_time,
                metadata_of=metadata_of,
            ):
                witness.report()
            raise e
        finally:
            witness.close()
            _db_io_witness.reset(witness_token)

    return wrapper


def is_exception_related_to_db(e: Exception) -> bool:
    """
    Returns True if the exception is related to the DB
    """

    import httpx
    from prisma.errors import PrismaError

    return isinstance(e, (PrismaError, httpx.TransportError))


_is_exception_related_to_db: Final = is_exception_related_to_db


async def _handle_logging_db_exception(
    e: Exception,
    func: Callable,
    kwargs: Mapping[str, object],
    args: tuple,
    start_time: datetime,
    end_time: datetime,
    metadata_of: Callable[[Mapping[str, object]], dict[str, str] | None],
) -> bool:
    from litellm.proxy.proxy_server import proxy_logging_obj

    # don't log this as a DB Service Failure, if the DB did not raise an exception
    if is_exception_related_to_db(e) is not True:
        return False

    try:
        await proxy_logging_obj.service_logging_obj.async_service_failure_hook(
            error=e,
            service=ServiceTypes.DB,
            call_type=func.__name__,
            parent_otel_span=kwargs.get("parent_otel_span"),
            duration=(end_time - start_time).total_seconds(),
            start_time=start_time,
            end_time=end_time,
            event_metadata=metadata_of(kwargs),
        )
    except Exception as hook_error:
        verbose_proxy_logger.debug("log_db_metrics: failure hook raised for %s: %s", func.__name__, hook_error)
    return True
