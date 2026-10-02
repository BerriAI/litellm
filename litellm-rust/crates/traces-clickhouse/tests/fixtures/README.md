# ClickHouse query fixtures

Run `cargo test -p litellm-traces-clickhouse --test queries --locked -- --test-threads=2` from `litellm-rust` with Docker running

Raw OTLP exports live in `crates/traces/tests/fixtures/query_*.json`. The seeded fixture decodes and normalizes them through `litellm_traces::decode_otlp` at test startup, then projects the decoded fields into ClickHouse columns. Team and key identities come from fixture setup rather than exporter claims. Root and child exports are inserted separately through the public insert API so materialized views process multiple blocks

`crates/traces/tests/fixtures/deeplite_auth_error.json` and `deeplite_swarm.json` were captured from Deeplite runs against the local proxy on 2026-10-02. The first contains a failed model call. The second contains successful model calls, searches, handoff attempts, and virtual filesystem writes. Credentials, workspace identifiers, and local user paths were redacted, and the protobuf exports were converted to OTLP JSON. Their round-trip tests check span identities, parent links, timestamps, durations, token counts, and statuses without pinning the provider's error wording

The swarm capture has handoff spans marked ERROR with `ParentCommand` exception events and a root with UNSET status. These are exported diagnostic statuses, which do not establish a failed execution. The tests preserve incoming statuses and check root status separately from the count of error spans, deriving both from the decoded export. They do not infer an execution outcome from exception text, framework names, successful model calls, or output presence. Framework-specific interpretation of control-flow exceptions belongs in the instrumentation integration

`spend_logs.jsonl` contains spend insert rows with millisecond timestamps, including two versions of one request. Replace this small placeholder dataset when the actual data is available. The query fixture applies production migrations, then removes TTL from its isolated database so fixed timestamps do not expire. Background merges are stopped so rollup aggregation and `FINAL` deduplication are exercised on unmerged data. Retention behavior stays covered by the migration tests

Curated SQL lives in `tests/queries/*.sql`. Each query has a matching `.expected.json` containing ordered result rows for `admin`, `team`, `key`, and `other_team` readers. Update the exports and expected results together. Add a named case in `tests/queries.rs` for each new query. Assertions compare only result data, excluding server statistics and execution timing

Typed query tests execute the production SQL through `litellm_storage_clickhouse::fetch` using contracts from `litellm-traces`. The fixture projection is test setup, so this suite covers the Rust decoder, normalization, inserts, schema, readers, and queries. Python ingress transformations, including payload truncation and exception-event fallback, remain covered by the Python tests
