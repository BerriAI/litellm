# LiteLLM Rust examples

Each example is a standalone binary Cargo package with its own `main` function. A link's URI scheme selects the example crate; the rest is a path relative to that crate.

## Proc macro

- [Open the entry point](proc-macro://src/main.rs#L1)
- [Open the derived data model](proc-macro://src/proof.rs#L3)
- [Open its Cargo manifest](proc-macro://Cargo.toml#L1)

`CompilerProof` derives Serde's `Serialize` and `Deserialize` procedural macros and references `litellm_core::Error`.

## In-repo dependency

- [Open the main function](error-type://src/main.rs#L3)
- [Open the report module](error-type://src/report.rs#L1)
- [Open its Cargo manifest](error-type://Cargo.toml#L1)

This example prints the fully qualified name of `litellm_core::Error`. No model or network request is involved.
