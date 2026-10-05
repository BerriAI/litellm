The trace API contract is owned by Rust. `litellm-rust/crates/traces/src/api.rs` defines public requests, responses, validation limits and defaults. `src/api/openapi.rs` defines paths, operation IDs, authentication and operation semantics. Domain response fields remain in their owning Rust types, and ClickHouse query-help types remain in `traces-clickhouse`

Run `uv run scripts/generate_trace_types.py` from the repository root to generate JSON Schema 2020-12, OpenAPI 3.1, Python validation models and route metadata. Run the same command with `--check` to detect drift. The committed OpenAPI document is `litellm/rust_bridge/trace/generated/openapi.json`

FastAPI imports generated request models and route metadata, invokes the Rust bridge, and merges the generated document into `/openapi.json`. Its route decorators do not own trace API documentation. The dashboard's `npm run gen:api` consumes that merged document. A future Rust HTTP server can consume the same request types and operation contract

| Operation | Contract |
| --- | --- |
| `GET /v1/traces` | Search summaries with `q`, bounded `page_size`, four sort fields and a continuation cursor |
| `GET /v1/traces/histogram` | Count the same search in bounded time buckets |
| `GET /v1/traces/values/{field}` | Return bounded top-K search suggestions |
| `GET /v1/traces/{id}` | Read summary and agent metadata by canonical ID |
| `GET /v1/traces/{id}/spans` | Traverse bounded pages of spans |
| `GET /v1/traces/{id}/spans/{span_id}` | Read raw span input, output and attributes |
| `GET /v1/traces/{id}/spans/{span_id}/error` | Traverse bounded diagnostic text |
| `POST /v1/traces/query` | Execute scoped, bounded ClickHouse SQL with native parameter binding |
| `GET /v1/traces/query/help` | Discover logical views, physical tables, examples and limits |
| `POST /v1/traces` | Receive the standard OTLP JSON or protobuf protocol |

The list, histogram and suggestions return a resolved `[start_ms,end_ms)` window and ingestion cutoff. Reuse those values to compare aggregates with the list. List cursors retain the window, query, scope and ordering. The cutoff excludes exports stamped later; buffered or distributed writes stamped before the cutoff can become visible later. It is not a database transaction snapshot. Collection pages use `data` and `next_cursor`; text pages retain their text-specific envelope. `trace_id` is the original OTLP identifier; `id` includes ownership and is the public read identifier. `root_status` describes the root span, while `has_error` reports any failed span

SQL exposes logical `traces`, `spans` and `calls` views under invoker row policies. Views summarize visible canonical spans. Curated user-only trace reads additionally require full ownership, so their membership can differ from SQL views. SQL pages are live and callers own keyset continuation. SQL span counters and token totals describe canonical normalized spans; curated call counts and token totals additionally resolve graph wrappers. These enrichment metrics have distinct SQL column names. Spend enrichment is best effort because replacing spend rows do not preserve historical versions

JSON read failures use RFC 9457 `application/problem+json` with a stable `code`. OTLP retains its protocol responses. Invalid or unknown query arguments are rejected

The generator pins datamodel-code-generator in inline dependency metadata and uses locked Schemars and Utoipa dependencies behind the optional Rust `schema` feature. These tools are not Python runtime dependencies. Templates preserve immutable Python collections and bounded types. Edit Rust or generation configuration, then regenerate; never edit generated contracts manually
