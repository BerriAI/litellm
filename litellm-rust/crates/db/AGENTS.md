# Database crate

Keep each static query in `queries/` and call it through a typed SQLx query function. Rust does not create or migrate the database

Run database tests against a local database already migrated by Prisma:

```sh
DATABASE_URL=postgresql://... cargo test -p litellm-db
```

After changing a query, refresh the checked query metadata from this directory:

```sh
SQLX_OFFLINE=false cargo sqlx prepare
```

Tests use transactions and leave them uncommitted so fixture rows roll back when each test finishes
