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
