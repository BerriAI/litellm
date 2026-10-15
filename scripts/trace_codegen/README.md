Run `uv run scripts/generate_trace_types.py` from the repository root to regenerate Python HTTP contracts. Run it with `--check` to verify the frozen Lens schemas and compare generated Python with committed files

Lens owns the Rust contracts. `scripts/lens_assets/source.json` pins the Lens repository revision and SHA-256 of each schema and development fixture. The gateway consumes these data files offline; it does not compile or maintain a second Rust tracing implementation

To update the pin, check out the intended Lens commit and run its `export-traces-schema` binary for the default, `--requests` and `--responses` modes, plus `export-traces-clickhouse-schema`, with the `schema` feature and locked dependencies. Import the consumed roots into the corresponding `scripts/lens_assets/schemas` group and copy fixtures from that same commit. Update the source revision and hashes together, then regenerate and run the gateway trace contract tests

The generator pins datamodel-code-generator through its inline dependency metadata. Neither it nor the schema exporter is a runtime dependency. The templates preserve frozen models, tuple conversion and bounded scalars. Request models retain each canonical schema's extra-field policy, including ignored unknown GET parameters and forbidden extra SQL body fields

The generated contracts live in `litellm/tracing/generated`. `requests.py` supplies gateway endpoint parameter models, `responses.py` supplies the SQL response, and `queries.py` validates the remote SQL envelope. Public request behavior changes need their own compatibility review
