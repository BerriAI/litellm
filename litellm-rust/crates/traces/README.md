# Trace storage foundation

This crate owns the initial ClickHouse schema, insert row encoding and parameterized read transport. Python adapters use `trace_schema_statements`, `trace_encode_rows` and `trace_query` from the native bridge. HTTP endpoints, ingestion and trace response assembly belong to the integration built on this foundation

The SQL files under `migrations/` are the canonical definitions for `otel_traces`, `agent_traces`, its materialized view, and `spend_logs`. `schema_statements` supplies the database and retention settings. Apply the returned statements in order using migration credentials. Setup is repeatable for a new installation; these initial `CREATE IF NOT EXISTS` statements do not upgrade an existing incompatible table. Future schema changes need explicit upgrade statements

`execute_read` uses ClickHouse typed placeholders and binds text, integer and string-array parameters separately from SQL. It obtains no privileges itself: callers must use a dedicated SELECT-only reader. The helper enforces a 15-second HTTP timeout, a 10-second query setting, 1,000 result rows and a 4 MiB response cap, and rejects errors embedded in HTTP 200 JSON responses. Large traces fail explicitly at those limits; span pagination is future work

Pass a shared `HttpClientPool` client using `ClientVariant::NoRedirect` and the host HTTP settings. The native bridge already does this. The SQL helper is internal infrastructure, not a public arbitrary-SQL endpoint

For self-hosted ClickHouse, install [config/reader.xml](config/reader.xml) as `/etc/clickhouse-server/users.d/litellm-traces-reader.xml` and set `LITELLM_TRACES_READER_PASSWORD` on the server. Change `default` in the three SELECT grants to the configured database and restrict the network range to the proxy network. The profile locks read-only mode, execution time, result rows and bytes, overflow behavior and memory limits. Managed ClickHouse deployments need equivalent grants and settings constraints

Keep reader, ingestion and migration credentials separate. Do not grant the reader write, backup, named-collection management or grant-option privileges, including through roles. `readonly=1` alone is insufficient for privileged accounts; see [query permissions](https://clickhouse.com/docs/concepts/features/configuration/settings/permissions-for-queries) and [settings constraints](https://clickhouse.com/docs/concepts/features/configuration/settings/constraints-on-settings)

Run `cargo test -p litellm-traces -- --test-threads=2` from `litellm-rust`. Testcontainers applies the canonical schema, checks span rollups and spend joins, and exercises reader permissions, parameter binding, credentials and response limits

`encode_rows` produces JSONEachRow and formats span nanoseconds and spend-log milliseconds as UTC DateTime64 strings without losing precision. The integration owns compression, batching and writes
