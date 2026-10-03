Run `uv run scripts/generate_trace_types.py` from the repository root to export Rust schemas and regenerate the Python trace contracts. Run the same command with `--check` to compare fresh output with the committed schemas and Python files

The script pins datamodel-code-generator in its inline dependency metadata. Rust uses the workspace's locked Schemars version through each owning crate's optional `schema` feature. Neither tool is a Python runtime dependency

Each crate exports its own roots using JSON Schema 2020-12. Request parameters use Schemars' deserialization contract. Trace views and query help use its serialization contract. Lens rows use their ClickHouse deserialization schemas, including quoted numbers and numeric boolean flags

The templates preserve tuple conversion, immutable tuple defaults, and bounded `ReadOnly` TypedDict fields. Pydantic models use the generator's frozen-model option and each schema's extra-field policy. ClickHouse numeric schemas select bounded, normalized Python scalar types through schema metadata consumed by the model template

Edit the owning Rust contract, schema annotation, or generation configuration, then regenerate. Never edit `litellm/rust_bridge/trace/generated/` manually. The SQL response envelope remains handwritten in `queries.py`
