Run `uv run scripts/generate_trace_types.py` from the repository root to export Rust schemas and regenerate the Python trace contracts. Run `uv run scripts/generate_trace_types.py --check` to compare fresh output with the committed schemas and Python files

The script pins datamodel-code-generator in its inline dependency metadata. Rust uses the workspace's locked Schemars version through each owning crate's optional `schema` feature. Neither tool is a Python runtime dependency

The `litellm-traces` Rust request types own the generated request models in `litellm/rust_bridge/trace/generated/requests.py`. The GET routes bind their query parameters directly to the generated models. GET request types allow unknown fields because existing clients' unknown query parameters are ignored. The SQL body model forbids extra fields

Each crate exports its own roots using JSON Schema 2020-12. Request parameters use Schemars' deserialization contract and carry only explicitly declared constraints. Their schemas skip the integer-bounds transform because it would add i64 bounds to `start_ms` and `end_ms`, narrowing what Python accepts, and replace `page_size`'s explicit 1..500 range with 0..65535. Trace views and query help use the serialization contract. Lens rows use their ClickHouse deserialization schemas, including quoted numbers and numeric boolean flags

The templates preserve tuple conversion, immutable tuple defaults, and bounded `ReadOnly` TypedDict fields. Pydantic models use the generator's frozen-model option and each schema's extra-field policy. ClickHouse numeric schemas select bounded, normalized Python scalar types through schema metadata consumed by the model template

Changing the public request contract requires a separate behavior-change PR

Edit the owning Rust contract, schema annotation, or generation configuration, then regenerate. Never edit `litellm/rust_bridge/trace/generated/` manually. The SQL response envelope remains handwritten in `queries.py`
