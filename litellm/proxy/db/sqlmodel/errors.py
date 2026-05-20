"""Prisma-compatible exception types backed by SQLAlchemy errors.

Existing call sites do ``except prisma.errors.UniqueViolationError`` and
similar. The shim raises *these* classes (re-exported from this module)
so call sites need not change. The Prisma package is no longer a runtime
dependency once the rip-and-replace lands; without these stand-ins,
``except prisma.errors.X`` would itself fail to import.
"""

from __future__ import annotations


class PrismaError(Exception):
    """Base class mirroring ``prisma.errors.PrismaError``.

    prisma-client-py's exception classes were Pydantic-ish: they accepted
    arbitrary keyword arguments to capture structured error metadata
    (table name, query, the upstream Prisma JSON envelope, etc.). We
    don't model that envelope, but we do accept and ignore arbitrary
    kwargs so legacy ``raise UniqueViolationError(data={"...": ...})``
    call sites and exception fixtures keep working.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args)
        self.data = kwargs.get("data")
        self._extra = kwargs


class DataError(PrismaError):
    """Generic data-layer error (e.g. constraint, type cast)."""


class UniqueViolationError(DataError):
    """Raised on UNIQUE constraint violations (Postgres SQLSTATE 23505)."""


class ForeignKeyViolationError(DataError):
    """Raised on FOREIGN KEY constraint violations (Postgres SQLSTATE 23503)."""


class RecordNotFoundError(DataError):
    """Raised when ``find_unique`` / ``update`` targets a non-existent row."""


class MissingRequiredValueError(DataError):
    """Raised when a non-null column is omitted from ``data=``."""


class TableNotFoundError(DataError):
    """Raised when a table referenced by the shim does not exist in the schema."""


class RawQueryError(DataError):
    """Raised when a ``query_raw`` / ``execute_raw`` statement fails."""


class ClientNotConnectedError(PrismaError):
    """Raised when the engine has been disposed but a query is attempted."""


class HTTPClientClosedError(ClientNotConnectedError):
    """Compatibility alias -- prisma-client-py distinguishes these."""


def map_sqlalchemy_error(exc: Exception) -> Exception:
    """Translate a SQLAlchemy ``IntegrityError`` (etc.) to a Prisma-compat error.

    The mapping is conservative: anything we cannot classify is wrapped in
    :class:`PrismaError` so callers' ``except PrismaError`` blocks still
    behave. Connection-layer errors are intentionally **not** mapped here
    -- they propagate untranslated so the existing reconnect logic in
    :class:`litellm.proxy.db.exception_handler.PrismaDBExceptionHandler`
    can recognise them.
    """
    from sqlalchemy.exc import (  # local import; SQLAlchemy is optional
        IntegrityError,
        NoResultFound,
        ProgrammingError,
        StatementError,
    )

    if isinstance(exc, NoResultFound):
        return RecordNotFoundError(str(exc))

    if isinstance(exc, IntegrityError):
        msg = str(exc.orig) if getattr(exc, "orig", None) is not None else str(exc)
        lowered = msg.lower()
        if "unique" in lowered or "duplicate key" in lowered:
            return UniqueViolationError(msg)
        if "foreign key" in lowered:
            return ForeignKeyViolationError(msg)
        if "not-null" in lowered or "violates not-null constraint" in lowered:
            return MissingRequiredValueError(msg)
        return DataError(msg)

    if isinstance(exc, ProgrammingError):
        msg = str(exc.orig) if getattr(exc, "orig", None) is not None else str(exc)
        if "does not exist" in msg.lower() and "relation" in msg.lower():
            return TableNotFoundError(msg)
        return RawQueryError(msg)

    if isinstance(exc, StatementError):
        return DataError(str(exc))

    return exc
