# Trace contract boundary

- `generated/` is output of `scripts/generate_trace_types.py` from the `litellm-traces` schemas. Change the Rust type and regenerate, never edit these files
- CI runs the generator with `--check` and fails on drift
- `storage.py` validates every native result against the generated response models before returning it
- Keep the trace methods in `_native.pyi` matching `litellm-rust/crates/python-bridge/src/routes/traces.rs`
- Native trace methods take scalar arguments. Moving them to the generated request types changes overflow errors, so it needs its own behavior-change PR
