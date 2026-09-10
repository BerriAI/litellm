# Proc macro app

This example is a standalone binary Cargo package with its own `main` function.

- [Open the entry point](proc-macro://src/main.rs#L1)
- [Open the derived data model](proc-macro://src/proof.rs#L3)
- [Open its Cargo manifest](proc-macro://Cargo.toml#L1)

`CompilerProof` derives Serde's `Serialize` and `Deserialize` procedural macros. The app also references `litellm_core::Error` from the in-repo LiteLLM Rust crate.
