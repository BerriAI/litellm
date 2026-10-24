# Database crate

Keep each static query in `queries/` and call it through a typed SQLx query function. Rust does not create or migrate the database

Database integration tests are ignored by default, but still compile in workspace checks. Run them explicitly against a local database already migrated by Prisma. Missing or invalid database configuration must fail the selected tests:

```sh
DATABASE_URL=postgresql://... cargo test -p litellm-db --locked -- --include-ignored
```

After changing a query, refresh the checked query metadata from this directory:

```sh
SQLX_OFFLINE=false cargo sqlx prepare
```

Tests use transactions and leave them uncommitted so fixture rows roll back when each test finishes
