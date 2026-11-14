"""Split each Postgres pool's connection budget between Prisma and the Rust core's SQLx pool.

Migration scaffolding. Prisma and SQLx share a database only while the Python gateway still runs Prisma
queries next to ported ones. These invariants hold until Prisma is gone:

- Operators set one `connection_limit` per pool (`database_connection_pool_limit`, a URL pin or
  `database_extra_connection_params`). Nothing an operator sets names Prisma, SQLx or Rust. The Rust share
  is carved out of that limit, never added on top, so a Rust pool does not grow the connection budget.
- The Rust share is the hard-coded `RUST_RESERVED_CONNECTIONS`. It is 0 until a ported query runs on the
  native pool, so Prisma keeps its whole limit, untouched, and no native pool opens.
- Only a positive integer limit is split. Any other value goes to Prisma as it is, which judges it the same
  way it did before the split existed.
- The split is Python's. It is about Prisma and Prisma lives only in the Python gateway, so this module is
  deleted with Prisma.
- The Rust core takes a plain `max_connections` and never learns about the split. That is already its final
  interface, since the Rust gateway will pass `database_connection_pool_limit` straight to SQLx.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

RUST_RESERVED_CONNECTIONS: Final = 0


@dataclass(frozen=True, slots=True)
class ConnectionBudget:
    total: int
    python: int
    rust: int

    @classmethod
    def split(cls, total: int, rust_reserved: int = RUST_RESERVED_CONNECTIONS) -> "ConnectionBudget":
        """Prisma keeps at least one connection, so a total at or below the reserve ends up over it."""
        return cls(total=total, python=max(1, total - rust_reserved), rust=rust_reserved)

    @classmethod
    def of(
        cls, connection_limit: str | float, rust_reserved: int = RUST_RESERVED_CONNECTIONS
    ) -> "ConnectionBudget | None":
        """The split of a positive integer limit, or None for any other value Prisma is left to judge."""
        digits: Final = str(connection_limit)
        if not (digits.isascii() and digits.isdigit()) or int(digits) == 0:
            return None
        return cls.split(int(digits), rust_reserved)

    def prisma_params(self, params: Mapping[str, str | int | float]) -> Mapping[str, str | int | float]:
        """``params`` with Prisma's share as its ``connection_limit``, untouched while the Rust pool takes none."""
        if self.rust == 0:
            return params
        return MappingProxyType({**params, "connection_limit": self.python})
