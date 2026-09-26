---
name: port-sql-query
description: Move one SQL query the LiteLLM proxy runs from Python into a shared query file, so Python keeps executing it through Prisma while litellm-db declares and compile-checks it with sqlx, and later switch it to Rust execution. Use when porting a query_raw, execute_raw or query_first call or a Prisma ORM call (find_many, update, upsert, ...), or when check_shared_sql_queries.py fails.
---

# Port a SQL query

During the migration Python owns migrations and keeps executing each query until it is switched. Porting moves the SQL into a shared file that Python runs and Rust declares. Read `litellm-rust/crates/db/AGENTS.md` first for how declarations, row structs, SQL rules and the `.sqlx` refresh work

## Pick the source shape

- SQL text passed to `query_raw`, `execute_raw`, `query_first` or a helper that forwards it: read [references/raw-sql.md](references/raw-sql.md)
- A Prisma ORM operation on a table (`.table.find_many(...)`, `db.litellm_teamtable.update(...)`): read [references/prisma-orm.md](references/prisma-orm.md), it ends in the steps below

## Steps

1. Write `litellm/proxy/db/queries/<domain>/<name>.sql`, with `<domain>` the table family (`keys`, `teams`, `autorouter`, ...). The file sits in the Python package so the wheel ships it
2. Load it in `<domain>/__init__.py` (create it for a new domain) as `<NAME>: Final = load(__name__, "<name>")`, with `<NAME>` the file stem in upper case
3. At the call site import the constant by name (`from litellm.proxy.db.queries.<domain> import <NAME>`, not the module) and pass it where the SQL text was, keeping the arguments in `$1`, `$2`, ... order
4. Declare it in `litellm-rust/crates/db/src/queries/<domain>.rs` (add `mod <domain>;` to `mod.rs` for a new domain), with a row struct if a caller reads the rows. Parameter types mirror what Python passes: `Option<T>` where Python may pass `None`, `&JsonValue` for `$n::jsonb`
5. Lower `INLINE_RAW_SQL_CEILING` or `ORM_CALL_CEILING` in `tests/code_coverage_tests/check_shared_sql_queries.py` by the number of calls moved. It wants the exact count, so a ceiling left too high fails too
6. Run `scripts/verify.sh` with `SQLX_DATABASE_URL` set to a scratch Postgres. It refreshes `.sqlx`, then runs clippy, the `postgres-tests` suite and the gate. Fix and rerun until it passes
7. Run the Python tests of the module you touched, preferring the database-backed ones (`tests/proxy_behavior/...`) with `DATABASE_URL` on the same scratch database. Mocked tests do not see SQL mistakes
8. Commit the `.sql` file, the Python change, the Rust declaration and `.sqlx` together

## Gotchas

- Never add sqlx `"col?"` or `"col!"` overrides to a shared file. Python reads them as the key names. LEFT JOIN columns are already nullable
- `SET LOCAL x = {value}` becomes `SET_STATEMENT_TIMEOUT` / `SET_LOCK_TIMEOUT` from `litellm.proxy.db.queries.transaction`, called with `str(value)`
- The gate counts a function that forwards one of its parameters as SQL as a raw call too, so moving SQL into a helper does not hide it
- SQL that cannot be static (DDL on computed identifiers such as partition names) stays inline with `# dynamic-sql-ok: <reason>` on the call. Anything with a finite set of shapes is not that case
- The gate also fails when a `.sql` file is not loaded by its domain package, not imported by any Python code, or not declared in Rust

## Switching a query to Rust execution (later)

- Make its declaration return the row struct instead of discarding it and drop the row's `dead_code` expectation
- Expose it through python-bridge, add a `postgres-tests` parity case against the Python path on the same seeded rows, and stage it in `litellm/rust_bridge/catalog.py`
- Queries that share a Prisma transaction switch together
