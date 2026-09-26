# litellm-db

- Every query is a static SQL file that sqlx checks at compile time against the Prisma-migrated schema
  - Prisma (`prisma migrate deploy`) owns the schema and migrations for now
  - Moving a query out of Python is a recipe in `.agent/skills/port-sql-query`
- Declaring a query
  - The SQL lives in `litellm/proxy/db/queries/<domain>/<name>.sql`: one statement, positional `$1` parameters
  - Declare it with one line in `declare_queries!` in `src/queries/<domain>.rs`, with typed parameters. A parameter type that does not match the schema fails to compile
  - A query whose rows a caller reads is declared as `name(params) -> Row => "file.sql";`, with `Row` hand-written next to it inside `declare_rows!`
  - Statements whose result nobody reads (writes, `set_config`) use the form without `-> Row`
- Row structs
  - `declare_rows!` gives every row the same derives (`Debug`, `Clone`, `PartialEq`), `schemars::JsonSchema` behind the `schema` feature, and a `dead_code` expectation that goes away once Rust executes the queries
  - `-> Row` expands to `query_file_as!`, so the struct is checked against the recorded columns: a wrong name, type, missing or extra field, or a nullable column in a non-`Option` field fails to compile
  - A `NOT NULL` column in an `Option` field still compiles, so keep `Option` for nullable columns only
  - The error points at the macro in `src/queries/mod.rs`, not at the field. The two types in the message name the field
  - Queries with the same columns share one struct (`BaselineObservationRow`)
  - Timestamps are chrono types (`NaiveDateTime` for `timestamp`, `DateTime<Utc>` for `timestamptz`) because schemars has no support for the `time` crate
- SQL rules
  - No `SELECT *` or `alias.*`: list the columns, so an additive migration does not change a query's shape. `tests/queries.rs` shows how to guard a list that has to cover every column of a table
  - Every output column needs a unique name that is a valid Rust identifier: alias expressions (`SELECT 1 AS found`) and never select the same alias twice
  - Session settings go through `SELECT set_config('statement_timeout', $1, true)` (the `transaction` domain), since `SET` cannot take a parameter
  - Keep SQL static
    - Optional filters: `($n::type IS NULL OR col = $n)`
    - IN lists: `= ANY($n::text[])`
    - Bulk writes: `UNNEST($1::text[], $2::float8[], ...)`, or `jsonb_to_recordset($1::jsonb) AS x(col type, ...)` for a JSON batch
    - A template with a few shapes becomes a few files, like `apply_baseline_session_corrections` and `apply_baseline_user_session_corrections`
- Offline cache
  - `.cargo/config.toml` forces `SQLX_OFFLINE=true`, so builds read this crate's `.sqlx` and never touch a database. Never unset it: builds would then depend on whatever database `DATABASE_URL` points at
  - Any new or edited `.sql` file (whitespace included) and any migration that touches a queried table needs a refresh. "`SQLX_OFFLINE=true` but there is no cached data for this query" means one was missed
  - Refresh from the repo root and commit `.sqlx` with the change: `SQLX_DATABASE_URL=<scratch postgres> make rust-sqlx-prepare`
    - The target runs `prisma migrate deploy` against `SQLX_DATABASE_URL` first, so never point it at a database you care about
    - A local one: `docker run -d --name litellm-sqlx -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=litellm -p 127.0.0.1:5544:5432 postgres:16`
    - It needs sqlx-cli 0.9.0: `cargo install sqlx-cli --version 0.9.0 --locked --no-default-features --features rustls,postgres`
  - A refresh that fails to compile means a migration and a query disagree, and the error says what the database rejected (`column t.team_alias does not exist`)
    - Renamed or dropped column: update the `.sql` file in the same change, then refresh
    - Changed type or nullability: update the parameter type or the row struct field
    - A migration that drops or renames a column released code still reads should be raised in review, not only patched here
  - rust-analyzer keeps showing the error on `declare_queries!` until the cache is refreshed
- Tests
  - Tests that need a database sit behind the `postgres-tests` feature and read `DATABASE_URL` of a migrated Postgres: `cargo test -p litellm-db --features postgres-tests`
  - Without the feature only tests that need no database run, so plain `cargo test` works anywhere
- Enforcement
  - The LiteLLM Rust DB workflow applies the real migrations, then runs `cargo sqlx prepare --check`, clippy with `postgres-tests,schema` and the `postgres-tests` suite
  - Clippy bans the unchecked `sqlx::query*` functions and `sqlx::raw_sql`
- Schema handshake
  - `build.rs` embeds the newest migration folder name as `REQUIRED_MIGRATION`
  - `schema_compatibility` reports `Behind` unless `_prisma_migrations` has that migration finished and not rolled back. Rust declines when the database is behind, and a database ahead of it is fine


## Data Modelling

- https://github.com/serde-rs/serde-rs.github.io/tree/master/_src
