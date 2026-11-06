# Lens gateway boundary

- Lens owns the canonical trace schemas and its service's normalization, storage, graph assembly and querying in `BerriAI/lens`
- This package owns the HTTP client, gateway response validation and asynchronous gateway spend export. When no Lens service is configured, its read-only Postgres spend-log adapter provides embedded agent traces using existing spend-log permissions. It does not ingest OTLP or implement Lens analysis
- Gateway endpoints preserve authenticated user/team scope, trace references, pagination and public error categories
- `generated/` comes from `scripts/generate_trace_types.py` using the Lens schemas pinned in `scripts/lens_assets/source.json`. Update the canonical Lens Rust contracts and import their schema outputs before regenerating. Never edit generated Python
- Keep the pinned schema and fixture checksums current. CI verifies assets and regenerated Python without a Rust trace implementation in this repository
- Independent gateway ClickHouse spend logging lives in `litellm/integrations/clickhouse`, using `litellm/rust_bridge/clickhouse.py` and the native spend writer
- OTLP upload routes only return setup guidance. Keep their JSON/protobuf error encoding without restoring ingestion or a native trace fallback
