# Prisma -> SQLModel migration -- Phase 1

This package is the foundation for migrating `litellm`'s proxy persistence
layer from Prisma (`prisma-client-py==0.11.0`) to SQLModel/SQLAlchemy. The
migration is multi-phase by necessity -- the proxy has ~1,680 Prisma client
call sites across ~147 production files plus ~177 test files, so any
"big-bang" cutover would be unreviewable and unsafe.

## What this Phase ships

| Artefact | Purpose |
|---|---|
| `schema_parser.py` | Tiny pure-Python parser for the subset of Prisma DSL actually used by `schema.prisma`. Used by the parity test and the generator. |
| `_generate.py` | Code generator that emits SQLModel class definitions from a parsed schema. Run it manually after schema changes. |
| `models.py` | SQLModel classes for **all 64 models** in `schema.prisma`. Hand-editable; the generator only seeds the file. |
| `tests/test_litellm/proxy/db/sqlmodel_orm/test_schema_parser.py` | Unit tests for the parser. |
| `tests/test_litellm/proxy/db/sqlmodel_orm/test_parity.py` | Parity test: every Prisma model has a matching SQLModel class with matching columns, nullability, primary keys, uniques, and indexes. Also asserts that the committed `models.py` is byte-identical to a fresh generator run. |

**Nothing in this package is wired into the runtime proxy yet.** Importing
the module has no effect on the existing Prisma-backed code paths -- the
generated classes simply sit alongside the Prisma client and are guarded
against drift by CI.

## Why a generator at all?

The schema is the source of truth and changes frequently. Hand-writing 64
SQLModel classes against a 1,378-line Prisma schema invites typos and
silent drift. The generator gives us one well-tested translation rule per
Prisma construct (`@id`, `@@index`, `String[]`, `@updatedAt`, etc.) and the
parity test catches any regression in either the schema or the generator.

When subsequent phases need to add SQLAlchemy-only behaviour (custom
relationships, hybrid properties, `Mapped[...]` annotations, etc.), edit
`models.py` by hand. The generator's output should still load and the
parity test should still pass; if they don't, the schema and the SQLModel
layer have diverged.

## Re-running the generator

```bash
uv run python -m litellm.proxy.db.sqlmodel._generate \
    --schema schema.prisma \
    --out litellm/proxy/db/sqlmodel/models.py
```

The parity test fails CI if a schema change isn't accompanied by a
regenerated `models.py`.

## Subsequent phases

The work below is the responsibility of follow-up PRs, in roughly this
order. Each phase is independently testable; do not bundle them.

1. **Session abstraction.** Introduce a thin `DBSession` interface that
   wraps the existing `prisma_client` today and a SQLAlchemy
   `AsyncSession` tomorrow. Land with zero behaviour change. This is the
   prerequisite for incrementally swapping call sites.
2. **CI: keep `schema.prisma` and `models.py` in sync.** Add a workflow
   that runs the parity test on every PR (the test already exists -- this
   step is just enabling it as a required check).
3. **Port the raw-SQL hotspots.** ~288 `query_raw` / `execute_raw` calls
   across ~37 files (concentrated in `spend_management_endpoints.py`,
   `db/create_views.py`, focus/cloudzero exporters). These are the
   easiest call sites to migrate -- the SQL is already there; we just
   swap the executor to a SQLAlchemy `session.execute(text(...))`.
4. **Migrate per-table call sites.** ~55 tables touched across ~1,680
   Prisma-client call sites. Parallelise by feature area
   (keys/teams/users -> spend/logs -> MCP/managed objects ->
   adaptive router/workflows). The session abstraction from phase 1 lets
   each call site flip independently.
5. **Replace the custom Prisma reliability layer.** The current
   `PrismaWrapper` (RDS IAM token rotation), `RoutingPrismaWrapper`
   (read/write split), and `PrismaDBExceptionHandler` (~10 distinct
   error type classifications) all need SQLAlchemy-native equivalents.
6. **Swap migrations to Alembic.** The current `litellm-proxy-extras`
   package bundles 123 Prisma migration files. Establish an Alembic
   baseline matching the live schema, with a documented "first-run
   after upgrade" path for existing deployments. The 10 / 123
   migrations that contain DML need careful translation; the rest are
   pure DDL and can be folded into the baseline for fresh installs.
7. **Tear out Prisma.** Remove `prisma==0.11.0` from `pyproject.toml`,
   `prisma generate` from all 7 Dockerfiles, the CI workflows that run
   it, the 3 `schema.prisma` copies (with their `check-schema-sync` and
   `sync-schema` workflows), and the `litellm-proxy-extras` migration
   bundle.

## Risks and gotchas surfaced during Phase 1

* **Reserved attribute names.** SQLModel/SQLAlchemy reserve `metadata`
  and `registry` on the mapped class. Several Prisma models have a
  `metadata Json` column. The generator emits these as Python attribute
  `metadata_` while keeping the on-disk column name `metadata` via
  `sa_column_kwargs={'name': 'metadata'}`. Migration of call sites must
  use `MyTable.metadata_` in Python.
* **`String[]` (Postgres array columns).** Prisma maps `String[]` to a
  Postgres `text[]`. The generator uses
  `sqlalchemy.dialects.postgresql.ARRAY(Text())`, which is
  Postgres-specific. SQLite-backed test environments will need a
  separate fixture path -- this is identical to the current Prisma
  situation (`prisma-client-py` on SQLite already requires manual JSON
  emulation).
* **`Json` columns default-text quoting.** Prisma's `@default("[]")` and
  `@default("{}")` emit `'[]'::jsonb` / `'{}'::jsonb` as the Postgres
  `DEFAULT`. The generator preserves both the Python `default_factory`
  *and* the `server_default` so migrated rows behave identically when
  the column is omitted from an INSERT.
* **`@updatedAt`.** Prisma updates the column from the client. The
  generator translates this to a SQLAlchemy `onupdate=lambda: ...
  utcnow()` so the behaviour persists when ported off Prisma.
* **`cuid()`.** Only `LiteLLM_CronJob.cronjob_id` uses it. The generator
  treats it as opaque-string-equivalent to `uuid()` (which is what every
  consumer already assumes).
* **Enums.** The single Prisma enum (`JobStatus`) is emitted as a Python
  `str`-Enum and the column is stored as `Text` to match what
  `prisma-client-py` already does on Postgres. A real `sa.Enum` can be
  introduced later if any call site benefits.
