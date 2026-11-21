# Lens gateway boundary

- Lens owns trace normalization, schemas, storage, graph assembly and querying in `BerriAI/lens`
- This package owns the HTTP client, gateway response validation and asynchronous gateway spend export. Keep the trace reader remote-only
- Gateway endpoints preserve authenticated user/team scope, trace references, pagination and public error categories
- `generated/` comes from `litellm/scripts/generate_trace_types.py` using the Lens schemas pinned in `litellm/scripts/lens_assets/source.json`. Update the canonical Lens Rust contracts and import their schema outputs before regenerating. Never edit generated Python
- Keep the pinned schema and fixture checksums current. CI verifies assets and regenerated Python without a Rust trace implementation in this repository
- Independent gateway ClickHouse spend logging lives in `litellm/integrations/clickhouse`, using `litellm/rust_bridge/clickhouse.py` and the native spend writer
- `POST /v1/traces` takes an agent's own spans with a gateway key and forwards them to Lens unchanged. Nothing is stored or parsed on the gateway. `POST /v1/logs` only returns setup guidance. Keep the JSON/protobuf error encoding and do not add a native trace fallback
