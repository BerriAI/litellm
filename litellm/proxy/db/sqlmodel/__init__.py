"""SQLModel-based ORM definitions for the LiteLLM proxy database.

This package is the foundation for migrating proxy persistence from Prisma to
SQLModel/SQLAlchemy. **Phase 1** (this module's current state) introduces:

* :mod:`schema_parser` -- a small ``schema.prisma`` parser used by the
  parity test (and by future code generators).
* :mod:`models` -- hand-maintained, generator-seeded SQLModel classes that
  mirror every model in the canonical ``schema.prisma``.
* A parity test (in ``tests/test_litellm/proxy/db/sqlmodel/``) that fails
  CI if the SQLModel definitions drift from the Prisma schema.

Nothing in this package is wired into the runtime proxy yet -- importing it
has no effect on existing Prisma-backed code paths. Subsequent phases will:

1. introduce a ``DBSession`` abstraction wrapping Prisma today and SQLAlchemy
   tomorrow,
2. port raw-SQL call sites (``query_raw`` / ``execute_raw``) onto SQLAlchemy,
3. migrate per-table call sites (~55 tables) behind the abstraction,
4. rebuild the ``PrismaWrapper`` / ``RoutingPrismaWrapper`` / exception
   classifier as SQLAlchemy-native components,
5. swap the migration tool from Prisma to Alembic with a baseline derived
   from the current schema state.

See ``litellm/proxy/db/sqlmodel/README.md`` for the full plan.
"""

from litellm.proxy.db.sqlmodel.models import ALL_MODELS

__all__ = ["ALL_MODELS"]
