# Repositories

`litellm-repositories` holds every static query the Rust side runs against the Postgres schema Prisma migrates, reads and writes alike, and the repository that is the only way to run each one. `litellm-db` underneath it holds the connection side only: pools, endpoints, errors, `observe`, `within`, `Statements` and `health_check`. It has no queries.

## Layout

One folder per repository, named after the object it reads or writes:

```
crates/repositories/
  .sqlx/                      # generated query metadata for the whole crate
  src/
    lib.rs                    # mod declarations and the public re-exports
    database.rs               # Database, Side, Session, Connections, Transaction: what every repository runs through
    inserted.rs               # Inserted: how an insert that can lose to a concurrent one ended
    raw_json.rs               # row helpers shared by more than one repository
    utc_timestamp.rs
    test_support.rs           # the database_pool fixture and shared fixture inserts
    jwt_key_mapping/
      mod.rs                  # mod lines and pub use of the public types, nothing else
      repository.rs           # the public repository
      queries.rs              # pub(super) query functions, row types and their database tests
      queries/
        find_mapped_token.sql
        mapped_claims_for_tokens.sql
    key/
      mod.rs
      repository.rs
      memory.rs               # a repository's own memory tier gets its own file
      queries.rs
      queries/...
    spend_log/
      mod.rs
      record.rs               # the typed log, delivery and write the spend log writer hands in
      repository.rs           # the window sums, and one transaction for the rows and their ledger deliveries
      queries.rs
      queries/...
```

## Rules

- Queries are private to their folder. Query functions and row types in `queries.rs` are `pub(super)`, so only that folder's `repository.rs` can call them, and not other repositories or other crates. Only a row type that a public repository method returns as is gets `pub` and a re-export from `mod.rs`, like `MappedClaim`
- Every query runs inside `Database::run` with a `Call`. `run` reports the operation, applies the socket deadline, and recycles the pools and retries once on a stale prepared plan. A query called outside it skips all three. The `Call`'s `call_type` and `table` are the names the Prisma version of the same read reported, so DB dashboards keep one series across both
- Take a connection with `Database::session(side)` and pass `session.executor()` to the query, so statements stay unnamed under PgBouncer. `Side::Read` prefers the replica. `Side::Writer` is for every write and for reads that must see the latest commit. A read that should see the latest commit but must keep working through a writer outage, like the model and credential reads a boot depends on, takes `Database::writer_preferred_session()` instead, which falls back to the replica only while the writer is marked down
- A write that must commit several queries together takes the session's connection with `Session::connection()`, opens the transaction on it, and passes `Session::names_statements()` to `Statements::new` for each query inside, as `spend_log` does. The whole transaction still runs inside one `Database::run`
- `repository.rs` holds the policy around the queries: the side, the `Call`, short-circuits that skip the round trip (an empty input list), fallbacks between queries, memory tiers, and shaping rows into the types the caller takes. Shaping that SQL can do, such as `COALESCE` defaults and joins, goes in the SQL so the row comes out ready
- Each query is one `.sql` file in its folder's `queries/`, read with `sqlx::query_file!`, `query_file_as!` or `query_file_scalar!`. The path is relative to the crate root, not the source file: `"src/key/queries/find_key_view.sql"`. The workspace `clippy.toml` bans the unchecked `sqlx::query` family outside test fixtures
- One exception to the `.sql` file rule: a query that differs only in which column it filters on, chosen from a short closed list, may be one runtime `sqlx::query`, `query_as` or `query_scalar` under `#[expect(clippy::disallowed_methods, reason = ...)]` instead of one file per column. The column must come from a `match` on an enum in the same function whose arms are `&'static str` SQL built with `concat!` from literals, never `format!` (sqlx only takes static SQL without `AssertSqlSafe`, which stays off limits), every value still goes through `.bind()`, and the `reason` names the enum. It gives up the compile-time check and the `.sqlx` entry, so the database tests must run the query once per arm. `spend_log::queries::window_spend` is the example
- A write returns the row it stored in the shape that object's read returns, from the same statement with `RETURNING`, so the host can cache it again. Cache writes and evictions stay with the host
- A write sets `created_at` and `updated_at` itself, as `NOW() AT TIME ZONE 'UTC'`. Prisma stamps both in UTC from its engine, there is no trigger behind `@updatedAt`, and a bare `CURRENT_TIMESTAMP` follows the session's time zone. The same goes for an id Prisma generates, such as `@default(uuid())`, which has no default in the database
- An insert that a concurrent one can beat uses `ON CONFLICT DO NOTHING` and returns `Inserted`. The row that won is read in a second statement, because the insert cannot see a row committed after it began
- An operation of more than one statement takes its connections through `Connections`, so its database test can run every step on one transaction
- A write of more than one statement runs in `Database::transaction`, inside `Database::run`. It commits when the steps return `Ok` and rolls back otherwise, so the retry `run` makes on a stale plan starts from nothing
- A write to a team's roster takes `pg_advisory_xact_lock(hashtext(team_id))` first, then any user row, the order the Python management endpoints lock in. The roster is a JSON column both sides rewrite, so skipping the lock loses updates and another order can deadlock
- Row types use `#[macro_rules_attribute::apply(crate::db_row)]` for the shared derives. Add `Serialize` by hand only when a row goes to the host unchanged, with `utc_timestamp` on its timestamp fields and `RawJson` for JSON columns the host parses itself
- A helper used by more than one repository lives at the crate root as `pub(crate)`. Anything used by one repository stays in its folder
- When two repositories need the same row, prefer each owning the query it needs. Raise a row type to `pub(crate)` only when both really run the same query

## Tests

- Database tests sit inline in `queries.rs` (or in `repository.rs` for policy that spans several queries, such as a transaction), use the `database_pool` fixture from `test_support`, are `#[ignore]`d with the reason `requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database`, and run inside a transaction they never commit, so their fixture rows roll back
- Policy that needs no database (memory tiers, row shaping) is tested inline in the file that holds it, without the database
- Run the database tests against a database Prisma has migrated:

```sh
DATABASE_URL=postgresql://... cargo test -p litellm-repositories --locked -- --include-ignored
```

## Query metadata

`.sqlx/` at the crate root holds the checked metadata for every query in every folder. The macros read it offline, so a build needs no database. After adding or changing a query, refresh it against a migrated Postgres container from the repository root:

```sh
make rust-sqlx-prepare
```

CI runs `cargo sqlx prepare --check` in this crate and fails when the metadata is stale

## Adding a repository

1. `src/<object>/` with `mod.rs`, `repository.rs`, `queries.rs` and `queries/*.sql`, following the rules above
2. `mod <object>;` and the `pub use` of the repository in `lib.rs`
3. `make rust-sqlx-prepare`, and commit the new `.sqlx` files
4. When Python calls it: a pyclass in `crates/python-bridge/src/db_repository/<object>.rs`, its stub in `litellm/rust_bridge/_native.pyi`, and a Python repository in `litellm/repositories/<object>_repository.py` that owns every call to it. Ruff's TID251 rule rejects importing the `Rust*Repository` classes anywhere else in `litellm/`
