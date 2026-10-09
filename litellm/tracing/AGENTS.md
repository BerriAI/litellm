# Lens gateway boundary

- Lens owns trace normalization, schemas, storage, graph assembly and querying in `BerriAI/lens`
- This package owns the HTTP client, gateway response validation and asynchronous gateway spend export. Keep the trace reader remote-only
- Gateway endpoints preserve authenticated user/team scope, trace references, pagination and public error categories
- `generated/` comes from `scripts/generate_trace_types.py` using the Lens schemas pinned in `scripts/lens_assets/source.json`. Update the canonical Lens Rust contracts and import their schema outputs before regenerating; never edit generated Python
- Keep the pinned schema and fixture checksums current. CI verifies assets and regenerated Python without a Rust trace implementation in this repository
- Independent gateway ClickHouse spend logging lives in `litellm/integrations/clickhouse`, using `litellm/rust_bridge/clickhouse.py` and the native spend writer
- OTLP upload routes only return setup guidance. Keep their JSON/protobuf error encoding without restoring ingestion or a native trace fallback
