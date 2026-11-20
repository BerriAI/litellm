"""Split each Postgres pool's connection budget between Prisma and the Rust core's SQLx pool.

Migration scaffolding. Prisma and SQLx share a database only while the Python gateway still runs Prisma
queries next to ported ones. These invariants hold until Prisma is gone:

- Operators set one `connection_limit` per pool (`database_connection_pool_limit`, a URL pin or
  `database_extra_connection_params`). Nothing an operator sets names Prisma, SQLx or Rust. The Rust share
  is carved out of that limit, never added on top, so Prisma and the Rust pool together never open more
  connections than the limit.
- Each pool needs a whole number limit of at least `MIN_CONNECTION_LIMIT`, since Prisma keeps at least one
  connection and auth reads keys through the Rust pool on a cache miss, so the Rust pool needs one too. The
  CLI refuses to start on any other limit rather than leave auth without a pool or open more than the limit.
- The Rust share is the hard-coded `RUST_RESERVED_CONNECTIONS`, cut down to what is left after Prisma's one.
  The writer and the reader are split separately, so each Rust pool takes its own pool's share.
- The split is Python's. It is about Prisma and Prisma lives only in the Python gateway, so this module is
  deleted with Prisma. The CLI hands the worker each Rust share through `RUST_WRITER_CONNECTIONS_ENV_VAR` and
  `RUST_READER_CONNECTIONS_ENV_VAR`, and a worker started without them reserves the full
  `RUST_RESERVED_CONNECTIONS` for each.
- The Rust core takes a plain `max_connections` per pool and never learns about the split. That is already
  its final interface, since the Rust gateway will pass `database_connection_pool_limit` straight to SQLx.
"""

import os
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

RUST_RESERVED_CONNECTIONS: Final = 2
MIN_CONNECTION_LIMIT: Final = 2
RUST_WRITER_CONNECTIONS_ENV_VAR: Final = "LITELLM_RUST_WRITER_DATABASE_CONNECTIONS"
RUST_READER_CONNECTIONS_ENV_VAR: Final = "LITELLM_RUST_READER_DATABASE_CONNECTIONS"
configure_pool_budget_from_settings: Final[ContextVar[bool]] = ContextVar(
    "configure_pool_budget_from_settings", default=False
)


class UnsplittableConnectionLimit(ValueError):
    """A pool's ``connection_limit`` leaves no room for both Prisma and the Rust pool."""

    def __init__(self, pool: str, connection_limit: object) -> None:
        super().__init__(
            f"{pool} connection_limit {connection_limit!r} must be a whole number of at least {MIN_CONNECTION_LIMIT}, "
            "one connection for Prisma and at least one for the Rust pool that reads keys for auth"
        )


@dataclass(frozen=True, slots=True)
class ConnectionBudget:
    total: int
    python: int
    rust: int

    @classmethod
    def split(cls, total: int, rust_reserved: int = RUST_RESERVED_CONNECTIONS) -> "ConnectionBudget":
        """Prisma keeps at least one connection, and the Rust pool takes what is left of its reserve."""
        rust: Final = max(0, min(rust_reserved, total - 1))
        return cls(total=total, python=total - rust, rust=rust)

    @classmethod
    def of(
        cls, pool: str, connection_limit: str | float, rust_reserved: int = RUST_RESERVED_CONNECTIONS
    ) -> "ConnectionBudget":
        """The split of a whole number limit of at least ``MIN_CONNECTION_LIMIT``, or UnsplittableConnectionLimit."""
        total: Final = _whole_number(connection_limit)
        if total is None or total < MIN_CONNECTION_LIMIT:
            raise UnsplittableConnectionLimit(pool, connection_limit)
        return cls.split(total, rust_reserved)

    def prisma_params(self, params: Mapping[str, str | int | float]) -> Mapping[str, str | int | float]:
        """``params`` with Prisma's share as its ``connection_limit``."""
        return MappingProxyType({**params, "connection_limit": self.python})


@dataclass(frozen=True, slots=True)
class RustConnections:
    writer: int
    reader: int


def rust_connections(environ: Mapping[str, str]) -> RustConnections:
    """The Rust pool sizes the CLI's split left this worker, or the full reserve for each the CLI set none for."""
    return RustConnections(
        writer=_share(environ, RUST_WRITER_CONNECTIONS_ENV_VAR), reader=_share(environ, RUST_READER_CONNECTIONS_ENV_VAR)
    )


def _capped_prisma_url(url: str, limit: int) -> str:
    parsed: Final = urlsplit(url)
    pairs: Final = tuple(parse_qsl(parsed.query, keep_blank_values=True))
    pins: Final = tuple(_whole_number(value) for key, value in pairs if key == "connection_limit")
    tighter_pins: Final = tuple(pin for pin in pins if pin is not None and 0 < pin <= limit)
    if len(pins) == len(tighter_pins) == 1:
        return url
    other: Final = tuple((key, value) for key, value in pairs if key != "connection_limit")
    return urlunsplit(parsed._replace(query=urlencode((*other, ("connection_limit", min((limit, *tighter_pins)))))))


def _configured_pool_env(limit: object, environ: Mapping[str, str]) -> Mapping[str, str]:
    writer_url: Final = environ.get("DATABASE_URL")
    if limit is None or not writer_url:
        return MappingProxyType({})
    if not isinstance(limit, (str, int, float)):
        raise UnsplittableConnectionLimit("database_connection_pool_limit", limit)
    budget: Final = ConnectionBudget.of("database_connection_pool_limit", limit)
    reader_url: Final = environ.get("DATABASE_URL_READ_REPLICA")
    reader: Final = (
        {
            "DATABASE_URL_READ_REPLICA": _capped_prisma_url(reader_url, budget.python),
            RUST_READER_CONNECTIONS_ENV_VAR: str(budget.rust),
        }
        if reader_url
        else {}
    )
    return MappingProxyType(
        {
            "DATABASE_URL": _capped_prisma_url(writer_url, budget.python),
            RUST_WRITER_CONNECTIONS_ENV_VAR: str(budget.rust),
            **reader,
        }
    )


@contextmanager
def with_configured_connection_budget(limit: object) -> Generator[None, None, None]:
    """Set the worker environment because Prisma and SQLx read it independently during startup."""
    updates: Final = _configured_pool_env(limit, os.environ)
    previous: Final = tuple((key, os.environ.get(key)) for key in updates)
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous:
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _share(environ: Mapping[str, str], name: str) -> int:
    share: Final = environ.get(name, "")
    return int(share) if share.isascii() and share.isdigit() else RUST_RESERVED_CONNECTIONS


def _whole_number(connection_limit: str | float) -> int | None:
    if isinstance(connection_limit, bool):
        return None
    if isinstance(connection_limit, float):
        return int(connection_limit) if connection_limit.is_integer() else None
    digits: Final = str(connection_limit)
    return int(digits) if digits.isascii() and digits.isdigit() else None
