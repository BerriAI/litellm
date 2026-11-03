# Database crate

Keep each static query in `queries/` and call it through a typed SQLx query function. Rust does not create or migrate the database

- Own reusable row projections in `src/` and export those consumed by other crates or integration tests
- Apply `#[macro_rules_attribute::apply(crate::db_row)]` to row types for shared derives, without adding serialization or schema derives automatically
- Apply `#[macro_rules_attribute::apply(crate::db_input)]` to write inputs, adding `Default` only where supported
- Keep function tests inline in their owning query module and reuse the strict database fixture in `test_support`
- Transaction-local setup in `begin` is the approved exception to static query files
- Keep write inputs separate from stored rows and keep test-only assertions out of public types
- `FromRow` supports runtime reads, but production static reads still use `query_file_as!` for compile-time checking

Database integration tests are ignored by default, but still compile in workspace checks. Run them explicitly against a local database already migrated by Prisma. Missing or invalid database configuration must fail the selected tests:

```sh
DATABASE_URL=postgresql://... cargo test -p litellm-db --locked -- --include-ignored
```

After changing a query, refresh the checked query metadata from this directory:

```sh
SQLX_OFFLINE=false cargo sqlx prepare
```

Tests use transactions and leave them uncommitted so fixture rows roll back when each test finishes

# Parity with the Python Prisma stack

Python and SQLx run side by side until every repository operation is native, so the SQLx pool must honor every database setting Python honors. `Endpoint` is resolved in Python from the final `DATABASE_URL` (`litellm/proxy/db/native_database.py`); Rust never reads `DATABASE_*` env vars or parses a Prisma URL. When `litellm/proxy/db/` gains a setting or behavior, add it here

| Python behavior | Where | Native status |
| --- | --- | --- |
| URL assembled from `DATABASE_*`, token minted, libpq TLS translated | `db_url_settings.py` | Inherited: the native endpoint reads the final URL |
| `connection_limit`, `pool_timeout`, `max_idle_connection_lifetime` | `proxy_cli.py`, `db_url_settings.py` | `PoolLimits`; `pool_timeout=0` (wait forever) falls back to SQLx's default |
| `connect_timeout`, `socket_timeout` | `proxy_cli.py` | Not carried over: SQLx has no counterpart, `acquire_timeout` bounds connecting |
| `options` (`statement_timeout`, `lock_timeout`, `idle_in_transaction_session_timeout`) | `proxy_cli.py` | `Endpoint::options`, sent as startup `-c` settings |
| `schema` | URL param | `search_path` startup setting when not `public`; a custom schema behind PgBouncer is rejected by PgBouncer |
| `pgbouncer=true`, `DATABASE_DISABLE_PREPARED_STATEMENTS` | `db_url_settings.py` | `statement_cache: false` |
| Rotating RDS IAM / Entra token | `token_auth.py`, `PrismaWrapper` | Python mints, `NativeDatabase.set_writer_password` swaps connect options |
| In-container PgBouncer | `pgbouncer.py` | Inherited: the pooled URL carries PgBouncer's credential |
| Startup connection budget | `postgres_connection_budget_message` | Counts the native writer pool |
| Read replica routing | `routing_prisma_wrapper.py` | Planned: optional reader pool with explicit routing |
| Error classification, `lock_timeout` on rollups | `exception_handler.py`, `rollup_lock_timeout.py` | Planned |
| `ServiceTypes.DB` events per round trip | `db_span.py`, `log_db_metrics.py` | Planned |
| Auth lookup gate, fail open on DB outage, reader choice per query | `db_lookup_gate.py`, `allow_requests_on_db_unavailable` | With each ported auth read |
| Health watchdog, engine reaper, drain tracker, engine recreation | `utils.py`, `query_engine_reaper.py`, `prisma_client.py` | Not needed: SQLx pools reconnect per connection and `close` waits for checked-out connections |
| Migrations, `check_migration`, views | `prisma_client.py`, `create_views.py` | Prisma keeps them until the SQLx cutover |
