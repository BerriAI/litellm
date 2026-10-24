# Database crate

Keep each static query in `queries/` and call it through a typed SQLx query function. Rust does not create or migrate the database

- Own reusable row projections in `src/` and export those consumed by other crates or integration tests
- Apply `#[macro_rules_attribute::apply(crate::db_row)]` to row types for shared derives, without adding serialization or schema derives automatically
- Keep write inputs separate from stored rows and keep test-only assertions out of public types
- `FromRow` supports runtime reads, but production static reads still use `query_file_as!` for compile-time checking

Run database tests against a local database already migrated by Prisma:

```sh
DATABASE_URL=postgresql://... cargo test -p litellm-db
```

After changing a query, refresh the checked query metadata from this directory:

```sh
SQLX_OFFLINE=false cargo sqlx prepare
```

Tests use transactions and leave them uncommitted so fixture rows roll back when each test finishes
