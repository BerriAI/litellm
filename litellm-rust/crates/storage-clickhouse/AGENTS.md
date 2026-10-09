# ClickHouse storage

`litellm-storage-clickhouse` exports `Storage`, a writer and bounded reader derived from one ClickHouse URL and database. It also exports bounded HTTP read and insert execution

The crate has no product tables, OTLP types, or named trace queries. `litellm-spend-clickhouse` supplies the gateway spend schema and row encoding

It also applies embedded SQLx migrations through the `_sqlx_migrations` ledger

Migration files are append-only, and changed applied files are rejected by their checksums. Startup migrations must be replay-safe schema changes because the runner records success after execution without dirty states or locks. Backfills belong in coordinated jobs outside proxy startup. The replay policy lives in `ClickHouseMigrate::apply`, `dirty_version`, and `lock`; a Keeper-backed or deploy-time runner changes only those methods
